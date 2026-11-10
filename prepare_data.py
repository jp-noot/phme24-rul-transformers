"""
Data Preparation CLI Script for NASA C-MAPSS Datasets.
Executes data extraction, RUL labeling, and target construction using utils/preprocessing.py.
"""

import argparse
import os
from utils.preprocessing import preprocess_cmapss_file, FD001_DROP_COLS


def main():
    parser = argparse.ArgumentParser(description="Preprocess NASA C-MAPSS dataset files.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="CMAPSS",
        help="Path to directory containing raw C-MAPSS text files.",
    )
    parser.add_argument(
        "--num_censored",
        type=int,
        default=0,
        help="Number of engine units to censor randomly (0 to disable censoring).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for censoring sampling.",
    )
    args = parser.parse_args()

    # Chemins des fichiers bruts et de sortie
    fd001_raw = os.path.join(args.data_dir, "train_FD001.txt")
    fd001_out = os.path.join(args.data_dir, "train_FD001_preprocessed.csv")

    fd004_raw = os.path.join(args.data_dir, "train_FD004.txt")
    fd004_out = os.path.join(args.data_dir, "train_FD004_preprocessed.csv")

    # Prétraitement FD001 (avec suppression des capteurs constants)
    if os.path.exists(fd001_raw):
        preprocess_cmapss_file(
            raw_txt_path=fd001_raw,
            output_csv_path=fd001_out,
            cols_to_drop=FD001_DROP_COLS,
            num_censored_units=args.num_censored,
            seed=args.seed,
        )

    # Prétraitement FD004
    if os.path.exists(fd004_raw):
        preprocess_cmapss_file(
            raw_txt_path=fd004_raw,
            output_csv_path=fd004_out,
            cols_to_drop=None,
            num_censored_units=args.num_censored,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()