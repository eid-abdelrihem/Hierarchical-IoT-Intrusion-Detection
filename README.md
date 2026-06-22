# Hierarchical IoT Intrusion Detection on CICIoT2023

**Paper:** *Adversarially Hardened Hierarchical Machine Learning for IoT Intrusion Detection on CICIoT2023*  
**Authors:** Eid Abd Elrihem, Ahmed Sherif, Ahmed Abdelhamed — School of Computing, Queen's University

---

## Overview

This repository contains the full source code, trained models, and experimental notebooks for a three-stage hierarchical intrusion detection system (IDS) evaluated on the CICIoT2023 benchmark.

Most IDS systems treat network traffic classification as a single flat multi-class problem. When the label space is large and heavily skewed — as in CICIoT2023, where some attack sub-types appear only ~1,200 times out of seven million flows — flat classifiers drop sharply on minority attacks. This project addresses that by breaking the classification task into three progressive stages, each focused on a narrower decision.

---

## Pipeline Architecture

```
Network Flow
     │
     ▼
┌─────────────────────┐
│  Stage 1: Binary    │  ──► Benign (dropped)
│  Gate (XGBoost)     │
└─────────────────────┘
     │ Attack
     ▼
┌─────────────────────┐
│  Stage 2: 7-Family  │  ── DDoS / DoS / Mirai / Recon / Spoofing / Web / Brute Force
│  Classifier         │
└─────────────────────┘
     │ Family ID
     ▼
┌─────────────────────┐
│  Stage 3: Sub-type  │  ── Dedicated model per family (33 sub-types total)
│  Specialist Models  │
└─────────────────────┘
     │
     ▼
Alert + SHAP Explanation
```

**Stage 1 — Binary Gate:** Separates attack traffic from benign. Uses all 46 original flow features with RobustScaler. XGBoost is selected as the gate after comparing Decision Tree, Random Forest, XGBoost, and MLP — all under the same 40-trial Optuna budget and identical data partitions.

**Stage 2 — Seven-Category Classifier:** Routes confirmed attacks into one of seven families using XGBoost with `multi:softprob` and class-specific probability multipliers tuned on a validation set.

**Stage 3 — Sub-type Specialist Models:** Each family with more than one sub-type gets a dedicated model trained with Borderline-SMOTE, pairwise interaction features (40 → 110 inputs), and 40-trial Optuna tuning. Brute Force (single sub-type) is mapped directly.

---

## Dataset

**CICIoT2023** — collected from a 105-device IoT topology. The raw corpus exceeds 46 million records.

- **Working sample:** ~7 million flows (stratified to preserve class proportions)
- **Features:** 46 pre-extracted statistical features (packet counts, byte stats, inter-arrival times, flag ratios)
- **Labels:** 34 classes — 1 benign + 33 attack sub-types in 7 families
- **Class imbalance:** ranges from 1,098,195 (benign) down to 1,252 (Uploading_Attack), an 877:1 ratio

---

## Results

### Key Numbers

| Stage | Task | Result |
|---|---|---|
| Binary Gate | Attack vs. Benign | F1-bin: 0.9931 · Miss: 0.81% · FAR: 2.99% · 0.44 µs/flow |
| Category | 7 Families | Macro F1: 0.9126 |
| Sub-types | 33 Classes | Macro F1: 0.9413 (full pipeline) |
| vs. Flat Baseline | Same test, same learner | +6.50 pp macro F1 (0.876 → 0.941) |
| vs. CICIoT2023 paper | 34-class task | +22.7 pp macro F1 (0.714 → 0.941) |

### Ablation — Hierarchical vs. Flat

Both systems use Optuna-tuned XGBoost on identical 1.4M test flows, so the gap reflects the hierarchical design itself.

| Metric | Flat XGBoost | Hierarchical | Δ |
|---|---|---|---|
| Macro F1 | 87.63% | **94.13%** | +6.50 pp |
| Accuracy | 97.22% | **98.52%** | +1.30 pp |
| Inference (CPU) | 18.51 µs | 19.85 µs | +7.3% |

The hierarchy uniquely corrects 23,816 flat-model errors; the flat model uniquely corrects only 5,692 — a 4.2× ratio.

### Adversarial Robustness (Binary Gate)

| Attack | Evasion |
|---|---|
| ZOO (black-box) | 0.00% |
| HopSkipJump (black-box) | 0.00% |
| Surrogate + PGD (transfer) | **0.08%** |
| Custom statistical shift | <0.01% |

After multi-attack retraining, PGD evasion drops to 0.00% at a small clean-miss trade-off (0.81% → 0.97%).

---

## Repository Structure

```
├── EDA_Feature_Engineering/
│   └── EDA_Feature_Engineering.ipynb       # Exploratory data analysis, feature engineering, class distribution
│
├── Binary classification/
│   └── binary_classification_Adverserial.ipynb  # Stage 1: binary gate + adversarial evaluation (ZOO, HopSkipJump, PGD, custom shift, retraining)
│
├── Multi Class Classification/
│   └── Multi_Class_7Attack1.ipynb          # Stage 2: seven-family classifier with threshold tuning and ceiling study
│
├── subclasses Classification/
│   └── SubClass_Classification_Experiments.ipynb  # Stage 3: per-family sub-type models with Borderline-SMOTE and Optuna
│
├── Ablation Study/
│   ├── 01_flat_34class_xgboost_ablation.ipynb     # Flat 34-class XGBoost baseline
│   ├── 02_hierarchical_end_to_end_ablation.ipynb  # End-to-end hierarchical pipeline evaluation
│   ├── 03_final_comparison_internal.ipynb         # Head-to-head comparison on identical test flows
│   ├── ablation_common.py                         # Shared utilities
│   ├── ablation_hierarchical_engine.py            # Hierarchical inference engine used in ablation
│   └── evaluate_internal_test_comparison.py       # Metric computation and row-level error analysis
│
├── Inference/
│   ├── inference_engine.py     # Full three-stage inference pipeline
│   ├── dashboard.py            # Streamlit prototype dashboard
│   ├── run_inference.py        # CLI entry point
│   ├── stream_simulator.py     # Simulates streaming traffic for demo
│   └── start.py                # Launcher
│
├── models/                     # Trained model files
├── results/                    # Benchmark results (CSV/JSON)
└── figures/                    # Plots used in the paper
```

---

## Preprocessing

Preprocessing is stage-specific:

- **Stage 1 (Binary):** All 46 original features, scaled with `RobustScaler` (median + IQR).
- **Stages 2 & 3 (Category + Sub-type):** Drop correlated features (Pearson |r| > 0.9), add 4 domain features → 40 columns. Stage 3 expands to 110 with pairwise interaction features.
- **Resampling:** Borderline-SMOTE applied inside each Stage 3 specialist model.

---

## Experimental Setup

- **Hardware:** NVIDIA RTX 5000 Ada Generation GPU, CUDA 12.6
- **Data split:** 80/20 stratified — ~5.6M training / ~1.4M test flows
- **Hyperparameter tuning:** Optuna, 40 trials per model, on an internal validation split

---

## Explainability

SHAP is applied to the binary gate over a 100,000-flow calibration sample. Top features: `IAT` (inter-arrival time, SHAP: 3.31), `rst_count`, and `Number` (packet count) — reflecting the volumetric and timing patterns that separate attack traffic from normal communication.

---

## Citation

If you use this code or dataset split in your work, please cite:

```
E. Abd Elrihem, A. Sherif, and A. Abdelhamed, "Adversarially Hardened Hierarchical
Machine Learning for IoT Intrusion Detection on CICIoT2023," 2026.
```

**Dataset:** E. C. P. Neto et al., "CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment," *Sensors*, vol. 23, no. 13, p. 5941, Jun. 2023.
