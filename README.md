<div align="center">

<img src="./artifacts/logo.png" alt="The Apex Predictors logo" width="240" />

# THE APEX PREDICTORS

## Interpretable ML for Salary Benchmarking

An anomaly detection and fair-market wage framework for developing economies utilizing macroeconomic indicators and XAI.

**Anthony Jairam** · **Chidera Ezenwaka** · **Elizabeth Halls** · **La Toya Paul** · **Terry-Jo Dass**

</div>

---

## Repository Layout

```
COMP6940-Project/
├── data/
│   ├── raw/
│   │   ├── job_salary_prediction_dataset.csv   # Primary salary dataset
│   │   └── Cost_of_Living_Index_by_Country_2024.csv
│   └── merged_data/
│       ├── merged.{csv,parquet}                # Cleaned, joined dataset
│       ├── train.parquet                       # 70 % split
│       ├── val.parquet                         # 15 % split
│       ├── test.parquet                        # 15 % split
│       ├── residuals.parquet
│       └── shap_values.parquet
├── artifacts/
│   ├── champion_model.pkl                      # Best XGBoost regressor
│   ├── xgb_preliminary.pkl
│   ├── scaler.pkl                              # Fitted StandardScaler
│   ├── encoder.pkl                             # OrdinalEncoder + TargetEncoder dict
│   └── figures/                                # Publication-ready PNGs
├── notebooks/part1/
│   ├── 01_data_loading_vizualization.ipynb
│   ├── 02_feature_egineering.ipynb
│   ├── 03_model_interpret.ipynb
│   ├── 04_shap_counterfactual.ipynb
│   └── 05_wage_fairness_analysis.ipynb
├── pyproject.toml
├── uv.lock
└── .python-version                             # Python 3.12
```

---

## Datasets


| Dataset                              | Source                                     | Records (approx.) |
| ------------------------------------ | ------------------------------------------ | ----------------- |
| Job Salary Prediction                |                                            |                   |
| Cost of Living Index by Country 2024 | [Numbeo / Kaggle](https://www.kaggle.com/) | 197 countries     |


After merging and cleaning, **25,065 rows** were dropped, leaving the final modelling dataset split 70 / 15 / 15 across train, validation, and test.

---

## Pipeline Overview


| #   | Notebook                              | Description                                                                                                                                          |
| --- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | `01_data_loading_vizualization.ipynb` | Load raw CSVs, inspect schema, exploratory data analysis and initial visualizations                                                                  |
| 2   | `02_feature_egineering.ipynb`         | Merge datasets, impute missing values, encode categoricals (OrdinalEncoder + TargetEncoder), scale numerics, and write train/val/test parquet splits |
| 3   | `03_model_interpret.ipynb`            | Train Linear Regression baseline, tuned XGBoost regressor, and MLP; experiment tracking via MLflow; select champion model                            |
| 4   | `04_shap_counterfactual.ipynb`        | Global and local SHAP explanations; DiCE-ML counterfactual scenarios; export SHAP values to `data/merged_data/shap_values.parquet`                   |
| 5   | `05_wage_fairness_analysis.ipynb`     | Wage anomaly detection; Fair Wage Index (FWI) using Global Skill Benchmark (GSB) and Cobb-Douglas Parity Benchmark (CDPB)                            |


Notebooks are designed to be run **in order** (1 → 5). Each notebook reads artifacts produced by its predecessors.

---

## Environment

### Requirements

- **Python:** 3.12 (see `.python-version`)
- **Package manager:** `[uv](https://github.com/astral-sh/uv)` (recommended) or `pip`


---

## Setup & Reproduction

### Option A: `uv` (recommended)

```bash
# Create virtual environment and install all dependencies from the lockfile
uv sync

# Launch Jupyter
uv run jupyter lab
```

### Option B: `pip`

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

jupyter lab
```

### Running the notebooks

Open each notebook in the `notebooks/part1/` directory and run cells top-to-bottom in order (01 → 05). All output paths are relative to the project root, so Jupyter must be launched from the project root.

### MLflow UI (optional)

Model training in Notebook 3 logs runs to a local MLflow tracking server. To inspect experiments:

```bash
uv run mlflow ui          # then open http://127.0.0.1:5000
```

---

## Loading Saved Artifacts

```python
import joblib

scaler  = joblib.load("artifacts/scaler.pkl")
encoder = joblib.load("artifacts/encoder.pkl")   # dict: {"ordinal": ..., "target": ...}
model   = joblib.load("artifacts/champion_model.pkl")
```