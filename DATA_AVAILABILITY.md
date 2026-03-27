# Data Availability & Repository Information

## GitHub Repository

**Public Repository URL:** https://github.com/Dani-Sh113/bde-prediction-ml


The complete code, data, and trained models have been deposited in a publicly accessible GitHub repository.

## Repository Contents

### 1. Data Preparation Script
**File:** `src/data_loader.py`
- Loads and validates datasets
- Handles missing values via mean imputation
- Splits data into training/test sets (80/20)
- Preprocesses external validation data

### 2. Feature Engineering & Correlation Analysis Script
**File:** `src/correlation_analysis.py`
- Calculates Pearson correlations between all 14 descriptors
- Identifies highly correlated features (|r| > 0.9)
- Removes redundant descriptors
- Generates high-quality correlation heatmaps

**Executable:** `python scripts/run_correlation_analysis.py`

### 3. Model Training & Selection Script
**File:** `src/model_training.py`

Uses **TPOT (Tree-based Pipeline Optimization Tool)** to automatically:
- Evaluate ~500 different machine learning pipeline configurations
- Run 10 generations × 50 population size with 10-fold cross-validation
- Select best-performing pipeline
- Generate training/test set visualizations (parity plots, residuals, metrics)

**Best Model Found:**
```
MinMaxScaler
    ↓
VarianceThreshold
    ↓
Nystroem (Feature Engineering)
    ↓
StackingEstimator (MLP Neural Network)
    ↓
LightGBM Regressor
```

**Performance:**
- Test R²: 0.9390
- Test MAE: 17.67 kJ/mol
- Test RMSE: 22.53 kJ/mol

**Executable:** `python scripts/run_model_training.py`

### 4. SHAP Analysis Script
**File:** `src/shap_analysis.py`

Conducts explainability analysis using SHAP (SHapley Additive exPlanations):
- Calculates SHAP values for all test predictions
- Ranks feature importance by mean absolute SHAP value
- Generates feature importance visualizations
- Creates SHAP dependence plots for top 3 features
- Produces feature contribution pie chart

**Top Features:**
1. C_GasteigerCharge (Partial charge on carbon)
2. X_GasteigerCharge (Partial charge on halogen)
3. Halogen_Z (Atomic number of halogen)

**Executable:** `python scripts/run_shap_analysis.py`

### 5. External Validation Script
**File:** `src/validation.py`

Validates trained model on truly external (unseen) data:
- Loads pre-trained model from `models/tpot_best_pipeline.pkl`
- Applies identical preprocessing as training data
- Generates predictions on ~100 external compounds
- Calculates comprehensive validation metrics
- Produces validation visualizations (parity plots, residuals, 4-panel analysis)
- Performs statistical tests (Shapiro-Wilk normality, Pearson/Spearman correlation)

**Executable:** `python scripts/run_validation.py`

### 6. Complete Pipeline Script
**File:** `scripts/run_full_pipeline.py`

Orchestrates all phases sequentially:
```
Correlation Analysis → Model Training → SHAP Analysis → External Validation
```

**Executable:** `python scripts/run_full_pipeline.py`

## Data Files

### Training Dataset
**File:** `data/processed/full_dataset_14_descriptors.csv`

- **Rows:** ~400 molecules with C-X bonds (X = halogen)
- **Columns:**
  - 5 metadata (Atoms, SMILES, BDE_kcal, BDE_kJ, Targeted_Halogen)
  - 14 molecular descriptors (MolWt, LogP, TPSA, NumRotBonds, HBD, FractionCsp3, NumHeavyAtoms, Halogen_Z, C_degree, C_hybridization, X_hybridization, C_GasteigerCharge, X_GasteigerCharge, X_EN)
- **Target:** BDE_kJ (Bond Dissociation Energy in kilojoules/mol)
- **Quality:** No missing values, complete feature set

**Usage:**
```python
from src.data_loader import load_full_dataset
df = load_full_dataset()  # Automatic path resolution
```

### External Validation Dataset
**File:** `data/processed/unseen_data.csv`

- **Rows:** ~100 molecules NOT in training set
- **Columns:** Same structure as training dataset
- **Purpose:** Independent evaluation of model generalization
- **Quality:** Same preprocessing as training data

**Usage:**
```python
from src.data_loader import load_and_prepare_external_data
X, y, features = load_and_prepare_external_data()
```

## Trained Models

### Pre-trained TPOT Model
**File:** `models/tpot_best_pipeline.pkl`

Serialized scikit-learn Pipeline ready for inference:
- File size: ~50-100 MB
- Format: Binary pickle format (joblib)
- No retraining required
- Consistent preprocessing embedded in pipeline

**Usage:**
```python
from src.utils import load_model
model = load_model()  # Automatically loads from models/tpot_best_pipeline.pkl
predictions = model.predict(X_new)
```

### Alternative: Generate Model
If pickle file is not available, regenerate by running:
```bash
python scripts/run_model_training.py
```
(Takes 2-4 hours depending on hardware)

## Key Module Descriptions

### src/config.py
Central configuration management:
- Feature names and counts
- File paths (automatic resolution)
- TPOT hyperparameters
- Threshold values
- Visualization settings

### src/utils.py
Reusable utilities:
- Plotting functions (heatmaps, metrics, residuals)
- Model I/O (loading/saving)
- Metrics calculation
- File operations

### src/data_loader.py
Data pipeline:
- CSV loading and validation
- Feature extraction
- Missing value imputation
- Train/test splitting
- Scaling and preprocessing

## Repository Structure

```
bde-prediction-ml/
├── README.md                              # Main documentation
├── IMPLEMENTATION_SUMMARY.md              # Reorganization details
├── requirements.txt                       # Python dependencies
├── setup.py                               # Package configuration
│
├── src/                                   # Python modules
│   ├── config.py                         # Configuration
│   ├── data_loader.py                    # Data loading
│   ├── correlation_analysis.py           # Feature correlation
│   ├── model_training.py                 # TPOT training
│   ├── shap_analysis.py                  # SHAP explainability
│   ├── validation.py                     # External validation
│   └── utils.py                          # Shared utilities
│
├── scripts/                               # Executable scripts
│   ├── run_correlation_analysis.py
│   ├── run_model_training.py
│   ├── run_shap_analysis.py
│   ├── run_validation.py
│   └── run_full_pipeline.py
│
├── data/
│   ├── processed/
│   │   ├── full_dataset_14_descriptors.csv      # Training data
│   │   └── unseen_data.csv                      # Validation data
│   ├── interim/
│   │   ├── highly_correlated_pairs.csv          # Generated
│   │   └── final_descriptor_list.csv            # Generated
│   └── README.md
│
├── models/
│   ├── tpot_best_pipeline.pkl                   # Trained model
│   ├── scaler.pkl                               # Data imputer
│   └── README.md
│
└── results/                                # Analysis outputs
    ├── correlation/
    ├── training/
    ├── shap/
    └── validation/
```

## Installation & Usage

### Setup
```bash
git clone https://github.com/Dani-Sh113/bde-prediction-ml
cd bde-prediction-ml
pip install -r requirements.txt
```

### Run Full Analysis
```bash
python scripts/run_full_pipeline.py
```

### Use Pre-trained Model for Predictions
```python
from src.utils import load_model
model = load_model()
predictions = model.predict(your_data)
```

## Dependencies

All required packages are listed in `requirements.txt`:
- pandas, numpy, scikit-learn
- matplotlib, seaborn (visualization)
- shap (explainability)
- tpot, lightgbm (models)
- joblib (serialization)

## Reproducibility

- Fixed random seed (42) throughout
- 10-fold cross-validation in TPOT
- Documented hyperparameters
- Saved metrics and predictions
- Version-controlled code

## Data Availability Statement

All code, data, and trained models required to reproduce the results are publicly available at:

**GitHub:** https://github.com/Dani-Sh113/bde-prediction-ml

**Permanent Archive (Zenodo):** [DOI: pending publication]

The repository includes:
1. Data preparation script (`src/data_loader.py`)
2. Feature engineering script (`src/correlation_analysis.py`)
3. Model training and selection script (`src/model_training.py`)
4. SHAP analysis script (`src/shap_analysis.py`)
5. Trained LightGBM model as `.pkl` file (`models/tpot_best_pipeline.pkl`)
6. External validation dataset (`data/processed/unseen_data.csv`)
7. Complete README with reproduction instructions

No proprietary software or data is required. All dependencies are open-source.

## Contact & Citation

If you use this code or data, please cite:

```bibtex
@software{bde_prediction_2024,
  title={BDE Prediction ML: Machine Learning for C-X Bond Dissociation Energy},
  author={Authors},
  year={2024},
  url={https://github.com/Dani-Sh113/bde-prediction-ml},
  doi={[Zenodo DOI]}
}
```

---

**Last Updated:** 2024-03-27
**Status:** Ready for peer review and publication
