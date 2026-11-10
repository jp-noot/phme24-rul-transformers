"""
Training Pipeline for DAST (Dual-Attention Sequence Transformer) on NASA C-MAPSS.
Supports censored temporal targets, custom masking for NaN targets, and CPU/GPU execution.
"""

from typing import Tuple
import argparse
import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import DAST_OR
from utils.preprocessing import create_sliding_windows, extract_statistical_features


class CMAPSSDataset(Dataset):
    """PyTorch Dataset wrapper for C-MAPSS sequence windows and targets."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        """
        Args:
            sequences: 3D numpy array of shape (N, Window_Size, Features).
            targets: 2D numpy array of shape (N, Target_Horizons).
        """
        super().__init__()
        # Strip categorical/metadata columns (last 3 columns: unit_number, time_cycle, RUL)
        self.x = torch.tensor(sequences[:, :, :-3], dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def masked_bce_loss(
    predictions: torch.Tensor, targets: torch.Tensor, criterion: nn.Module
) -> Tuple[torch.Tensor, float, float]:
    """
    Compute Binary Cross Entropy loss and accuracy while masking out NaN targets (censored data).

    Args:
        predictions: Model output probabilities (N, Horizon_Size).
        targets: Target tensor containing binary values or NaNs.
        criterion: PyTorch BCE loss module (reduction='sum').

    Returns:
        loss: Torch scalar loss with autograd enabled.
        correct_count: Number of accurate binary predictions.
        total_valid_count: Total non-NaN target count evaluated.
    """
    # Create mask for valid (non-NaN) targets
    valid_mask = ~torch.isnan(targets)
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=predictions.device, requires_grad=True), 0.0, 0.0

    valid_preds = predictions[valid_mask]
    valid_targets = targets[valid_mask]

    loss = criterion(valid_preds, valid_targets)

    # Binary accuracy on non-NaN elements
    binary_preds = (valid_preds >= 0.5).float()
    correct_count = (binary_preds == valid_targets).sum().item()
    total_valid_count = valid_mask.sum().item()

    return loss, correct_count, total_valid_count


def train_dast(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
    device: torch.device,
    save_dir: str = "checkpoints",
    experiment_id: str = "exp0",
) -> pd.DataFrame:
    """
    Main training and validation loop for DAST architecture.
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)

    history = {
        "Epoch": [],
        "Loss_train": [],
        "Accuracy_train": [],
        "Loss_val": [],
        "Accuracy_val": [],
    }

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)

            loss, correct, total = masked_bce_loss(preds, batch_y, criterion)
            
            if total > 0:
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                train_correct += correct
                train_total += total

        train_acc = train_correct / max(train_total, 1.0)
        train_loss_avg = train_loss / max(train_total, 1.0)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0.0, 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)

                loss, correct, total = masked_bce_loss(preds, batch_y, criterion)
                val_loss += loss.item()
                val_correct += correct
                val_total += total

        val_acc = val_correct / max(val_total, 1.0)
        val_loss_avg = val_loss / max(val_total, 1.0)

        # Record metrics
        history["Epoch"].append(epoch)
        history["Loss_train"].append(train_loss_avg)
        history["Accuracy_train"].append(train_acc)
        history["Loss_val"].append(val_loss_avg)
        history["Accuracy_val"].append(val_acc)

        print(
            f"Epoch [{epoch + 1:03d}/{epochs:03d}] | "
            f"Train Loss: {train_loss_avg:.4f} - Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss_avg:.4f} - Val Acc: {val_acc:.4f}"
        )

        # Save best model checkpoint
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            checkpoint_path = os.path.join(save_dir, f"best_model_{experiment_id}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss_avg,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )

    df_history = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, "results", f"history_{experiment_id}.csv")
    df_history.to_csv(csv_path, index=False)

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time / 60:.2f} minutes.")
    return df_history


def main():
    parser = argparse.ArgumentParser(description="Train DAST model on C-MAPSS dataset.")
    parser.add_argument("--data_path", type=str, default="CMAPSS/train_FD001_preprocessed.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--exp_id", type=str, default="FD001_exp1")
    
    # Model Hyperparameters lightweight configuration for CPU execution
    parser.add_argument("--dim_model", type=int, default=32, help="Hidden dimension for attention blocks")
    parser.add_argument("--n_heads", type=int, default=2, help="Number of multi-head attention heads")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training pipeline on device: {device}")

    # Load preprocessed data
    df = pd.read_csv(args.data_path)

    # Train / Validation Split (80/20 by unit_number)
    unique_units = df["unit_number"].unique()
    train_units, val_units = train_test_split(unique_units, test_size=0.2, random_state=42)

    train_df = df[df["unit_number"].isin(train_units)].copy()
    val_df = df[df["unit_number"].isin(val_units)].copy()

    quanti_cols = [
        "operational_setting_1", "operational_setting_2", "sensor_2", "sensor_3", 
        "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12", 
        "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]
    quali_cols = ["unit_number", "time_cycle", "rul"]
    target_cols = [f"target_{step * 10}" for step in range(1, 14)]

    # Generate sliding window arrays
    X_train, y_train, mean, std = create_sliding_windows(
        train_df, quali_cols, quanti_cols, target_cols, args.window_size
    )
    X_val, y_val, _, _ = create_sliding_windows(
        val_df, quali_cols, quanti_cols, target_cols, args.window_size, mean=mean, std=std
    )

    # Statistical feature enrichment
    X_train_enhanced = extract_statistical_features(X_train, args.window_size, len(quanti_cols))
    X_val_enhanced = extract_statistical_features(X_val, args.window_size, len(quanti_cols))

    # PyTorch DataLoaders
    train_dataset = CMAPSSDataset(X_train_enhanced, y_train)
    val_dataset = CMAPSSDataset(X_val_enhanced, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate Model (Lightweight DAST for CPU)
    model = DAST_OR(
        dim_val_s=args.dim_model,
        dim_attn_s=args.dim_model,
        dim_val_t=args.dim_model,
        dim_attn_t=args.dim_model,
        dim_val=args.dim_model,
        dim_attn=args.dim_model,
        time_step=args.window_size + 2,
        input_size=len(quanti_cols),
        dec_seq_len=10,
        out_seq_len=len(target_cols),
        n_decoder_layers=1,
        n_encoder_layers=2,
        n_heads=args.n_heads,
        dropout=0.1,
    ).to(device)

    criterion = nn.BCELoss(reduction="sum")
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # Start training
    train_dast(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
        device=device,
        experiment_id=args.exp_id,
    )


if __name__ == "__main__":
    main()