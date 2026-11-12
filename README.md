# PHME 2024, Dual-Attention Transformers for RUL Estimation under Data Censoring (`DAST-OR`)

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Award](https://img.shields.io/badge/PHME%202024-Best%20Paper%20Award-gold)](#publications--citations)

Official PyTorch implementation of **DAST-OR** (Dual Aspect Self-Attention based on Transformer utilizing Ordinal Regression) for Remaining Useful Life (RUL) estimation on the NASA C-MAPSS dataset under data (right)-censoring conditions.

This repository accompanies our award-winning work presented at **PHME 2024** (https://papers.phmsociety.org/index.php/phme/article/view/3974) and expanded in **IJPHM** (https://papers.phmsociety.org/index.php/ijphm/article/view/4260).

---

## 💡 Key Contributions & Architecture

- **`DAST_OR` (Core Contribution):** A novel Dual Aspect Self-Attention based on Transformer architecture utilizing **Ordinal Regression (OR)** to handle data censoring. Predicts cumulative failure probabilities across discrete temporal horizons ($\text{target}_{10}$ to $\text{target}_{130}$) with a sigmoid-bounded output and masked Loss computation.
- **`DAST` (Refactored Baseline):** A clean, bug-fixed PyTorch implementation of the standard baseline Dual Aspect Self-Attention based on Transformer, featuring proper `nn.ModuleList` submodules, optimized tensor operations, and CPU/GPU memory efficiency.

---

## 📁 Repository Structure

```text
phme24-rul-transformers/
├── CMAPSS/                      # NASA dataset directory (ignored by git)
├── models/
│   ├── __init__.py
│   ├── dast_network.py          # Core DAST & DAST_OR architectures
│   └── dast_utils.py            # Attention blocks & Positional Encodings
├── utils/
│   ├── __init__.py
│   └── preprocessing.py        # Sliding window, Z-score scaling & statistical features
├── prepare_data.py              # CLI data processing tool
├── train.py                     # Training script with vectorized loss & PyTorch DataLoaders
├── .gitignore                   # Excludes raw data (.txt/.csv) & checkpoints (.pt)
├── requirements.txt
└── README.md
```

---

## 📦 Dataset Setup (NASA C-MAPSS)

1. Download the `.zip` archive containing all sub-datasets (FD001–FD004) from the official [NASA Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data).
2. Extract the archive contents into the `CMAPSS/` folder at the root of the project:
   ```text
   phme24-rul-transformers/
   └── CMAPSS/
       ├── train_FD001.txt
       ├── test_FD001.txt
       ├── RUL_FD001.txt
       ├── train_FD004.txt
       └── ...
   ```
3. Run the automated data processing CLI:
   ```bash
   python prepare_data.py --num_censored 0
   ```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone [https://github.com/jp-noot/phme24-rul-transformers.git](https://github.com/jp-noot/phme24-rul-transformers.git)
cd phme24-rul-transformers
pip install -r requirements.txt
```

### 2. Model Training (`DAST_OR`)

To launch model training on CPU or GPU with lightweight parameters:

```bash
python train.py --epochs 30 --batch_size 128 --dim_model 32 --n_heads 2 --exp_id FD001_cpu
```

---

## 📜 Publications & Citations

If you use this codebase or model architecture in your research, please cite our papers:

### PHM Society European Conference 2024 (Best Paper Award)
> **Title:** *LSTM and Transformers based methods for Remaining Useful Life Prediction considering Censored Data *  
> **Conference:** 8th European Conference of the PHM Society (PHME 2024)

### International Journal of Prognostics and Health Management (IJPHM) &rarr Extended version of the previous article 
> **Title:** *LSTM and Transformers based methods for Remaining Useful Life Prediction considering Censored Data *  
> **Journal:** International Journal of Prognostics and Health Management (IJPHM)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
