# BDE Prediction ML: C-X Bond Dissociation Energy Prediction using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TPOT](https://img.shields.io/badge/AutoML-TPOT-brightgreen.svg)](http://epistasislab.github.io/tpot/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive machine learning pipeline for predicting **C-X Bond Dissociation Energy (BDE)** values, where X is a halogen (F, Cl, Br, or I). This project uses automated machine learning (TPOT), feature analysis (SHAP), and external validation to deliver robust predictive models.

## Overview

**Bond Dissociation Energy** is the amount of energy required to break a chemical bond. For C-X bonds (where X is a halogen), BDE is a critical property in:
- Organic chemistry reaction planning
- Computational chemistry validation
- Drug design and medicinal chemistry
- Environmental fate prediction

This project provides:
- ✅ **Modular Python codebase** split into logical analysis phases
- ✅ **Automated model selection** using TPOT
- ✅ **Feature importance analysis** using SHAP
- ✅ **Pre-trained models** ready for deployment
- ✅ **Comprehensive documentation** and results
- ✅ **Easy to extend** with new data or methods

## Project Structure

```
bde-prediction-ml/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.py                   # Package configuration
│
├── data/                      # Data directory
│   ├── processed/            # Training & validation datasets
│   ├── interim/              # Intermediate processing outputs
│   └── README.md             # Data documentation
│
├── models/                    # Pre-trained models
│   ├── tpot_best_pipeline.pkl # Trained TPOT model (binary)
│   ├── scaler.pkl            # Data imputer/scaler (binary)
│   └── README.md             # Model documentation
│
├── src/                       # Source code modules
│   ├── config.py             # Configuration & constants
│   ├── data_loader.py        # Data loading utilities
│   ├── correlation_analysis.py # Feature correlation analysis
│   ├── model_training.py     # TPOT model training
│   ├── shap_analysis.py      # SHAP feature importance
│   ├── validation.py         # External validation
│   └── utils.py              # Shared utility functions
│
├── scripts/                   # Execution scripts
│   ├── run_correlation_analysis.py    # Phase 1
│   ├── run_model_training.py          # Phase 2
│   ├── run_shap_analysis.py           # Phase 3
│   ├── run_validation.py              # Phase 4
│   └── run_full_pipeline.py           # All phases
│
├── notebooks/                 # Jupyter notebooks (optional)
│   ├── 01_correlation_analysis.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_shap_analysis.ipynb
│   └── 04_external_validation.ipynb
│
└── results/                   # Output results
    ├── correlation/          # Feature correlation visualizations
    ├── training/            # Training plots & metrics
    ├── shap/                # Feature importance analysis
    └── validation/          # External validation results
```

## Quick Start

### Installation

```bash
# Clone or download the repository
cd bde-prediction-ml

# Install dependencies
pip install -r requirements.txt
```

### Using Pre-trained Model (No Training Required!)

```python
import joblib
import numpy as np
from src.data_loader import load_and_prepare_external_data

# Load the pre-trained model
model = joblib.load('models/tpot_best_pipeline.pkl')

# Load your data (must have 14 molecular descriptors)
X_new, y_new, feature_names = load_and_prepare_external_data('path/to/your/data.csv')

# Make predictions
predictions = model.predict(X_new)
print(f"Predicted BDE values: {predictions}")
```

### Running the Full Pipeline

```bash
# Run all phases in sequence (correlation → training → analysis → validation)
python scripts/run_full_pipeline.py
```

### Running Individual Phases

```bash
# Phase 1: Feature correlation analysis
python scripts/run_correlation_analysis.py

# Phase 2: Model training with TPOT
python scripts/run_model_training.py

# Phase 3: SHAP feature importance analysis
python scripts/run_shap_analysis.py

# Phase 4: External dataset validation
python scripts/run_validation.py
```

## Model Architecture

The best-performing model discovered by TPOT uses:
- **MinMaxScaler** - Feature normalization to [0, 1] range
- **Nystroem kernel approximation** - Feature engineering
- **MLP Neural Network** - Initial predictions
- **LightGBM Regressor** - Final predictions with gradient boosting

## Performance

### Training Set
- **R² Score:** 0.9760
- **MAE:** 11.18 kJ/mol
- **RMSE:** 14.09 kJ/mol

### Test Set
- **R² Score:** 0.9390
- **MAE:** 17.67 kJ/mol
- **RMSE:** 22.53 kJ/mol

### External Validation Set
- **R² Score:** 0.94+
- **Predictions on unseen data show excellent generalization**

## Dataset

The project uses **14 molecular descriptors** to predict C-X bond dissociation energy:

| Descriptor | Description | Example Value |
|---|---|---|
| MolWt | Molecular weight | 100-200 |
| LogP | Lipophilicity | -2 to 5 |
| TPSA | Topological Polar Surface Area | 0-200 Å² |
| NumRotBonds | Rotatable bonds | 0-15 |
| HBD | Hydrogen bond donors | 0-5 |
| FractionCsp3 | sp3 carbon fraction | 0-1 |
| NumHeavyAtoms | Non-hydrogen atoms | 5-50 |
| Halogen_Z | Halogen atomic number | 9-53 |
| C_degree | Carbon hybridization degree | 2-4 |
| C_hybridization | Carbon hybridization type | 2-4 |
| X_hybridization | Halogen hybridization type | 2-4 |
| C_GasteigerCharge | Partial charge on carbon | -1 to 1 |
| X_GasteigerCharge | Partial charge on halogen | -1 to 1 |
| X_EN | Halogen electronegativity | 2.6-4.0 |

**Target Variable:** BDE_kJ (Bond Dissociation Energy in kJ/mol)

See `data/README.md` for detailed data schema and preprocessing steps.

## Feature Importance (SHAP Analysis)

Top 5 most important features for BDE prediction:

1. **C_GasteigerCharge** - Partial charge on carbon atom
2. **X_GasteigerCharge** - Partial charge on halogen
3. **Halogen_Z** - Atomic number of halogen
4. **MolWt** - Molecular weight
5. **C_hybridization** - Hybridization state of carbon

See `results/shap/` for detailed SHAP visualizations and dependence plots.

## Usage Examples

### Example 1: Predict BDE for a single molecule

```python
import joblib
import pandas as pd
import numpy as np

# Load model
model = joblib.load('models/tpot_best_pipeline.pkl')

# Create feature vector (must be 14 values in the correct order!)
features = np.array([[
    130.0,      # MolWt (Molecular weight)
    2.5,        # LogP
    20.0,       # TPSA
    2,          # NumRotBonds
    0,          # HBD
    0.8,        # FractionCsp3
    10,         # NumHeavyAtoms
    9,          # Halogen_Z (F)
    4,          # C_degree
    4,          # C_hybridization
    3,          # X_hybridization
    0.1,        # C_GasteigerCharge
    -0.3,       # X_GasteigerCharge
    3.98        # X_EN (Electronegativity of F)
]])

# Predict BDE
bde_prediction = model.predict(features)
print(f"Predicted BDE: {bde_prediction[0]:.2f} kJ/mol")
```

### Example 2: Batch predictions from CSV

```python
import joblib
import pandas as pd

# Load model and data
model = joblib.load('models/tpot_best_pipeline.pkl')
df = pd.read_csv('data/processed/unseen_data.csv')

# Prepare features (drop non-feature columns)
non_feature_cols = ['Atoms', 'BDE_kcal', 'BDE_kJ', 'SMILES', 'Targeted_Halogen', 'Unnamed: 0']
X = df.drop(columns=non_feature_cols, errors='ignore')

# Predict
predictions = model.predict(X.values)
df['Predicted_BDE'] = predictions
df.to_csv('predictions_output.csv', index=False)
```

## Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0
tpot>=0.12.0
lightgbm>=3.3.0
joblib>=1.0.0
```

See `requirements.txt` for the complete list.

## reproducibility

- **Random seed:** 42 (used throughout for reproducibility)
- **Train/test split:** 80/20 with random_state=42
- **Cross-validation:** 10-fold CV during model selection
- **TPOT parameters:**
  - Generations: 10
  - Population size: 50
  - Total models evaluated: ~500

All TPOT model configurations, metrics, and visualizations are saved in `results/training/`.

## File Outputs

### After Running Full Pipeline

```
results/
├── correlation/
│   ├── correlation_heatmap_all_descriptors.png       # Before removing correlated features
│   ├── correlation_heatmap_reduced_descriptors.png   # After removing correlated features
│   └── highly_correlated_pairs.csv                   # Correlation pairs (r > 0.9)
│
├── training/
│   ├── parity_plots.png                              # Train vs test predictions
│   ├── combined_parity_plot.png                      # Combined visualization
│   ├── residual_analysis.png                         # 4-panel residual plots
│   ├── metrics_comparison.png                        # Performance comparison
│   └── training_metrics.csv                          # Detailed metrics
│
├── shap/
│   ├── shap_feature_importance_bar.png               # Feature importance ranking
│   ├── shap_summary_beeswarm.png                     # SHAP beeswarm plot
│   ├── shap_dependence_*.png                         # Top 3 feature dependencies
│   ├── shap_contribution_pie_chart.png               # Feature contributions
│   ├── shap_feature_importance_analysis.csv          # Detailed importance metrics
│   └── shap_values_detailed.csv                      # Raw SHAP values
│
└── validation/
    ├── validation_parity_plot.png                    # External validation parity plot
    ├── validation_residuals.png                      # Residual analysis
    ├── validation_comprehensive.png                  # 4-panel comprehensive analysis
    ├── validation_predictions.csv                    # Predictions on external data
    └── validation_summary.csv                        # Validation metrics
```

## Publication & Citation

If you use this project in research, please cite:

```bibtex
@software{bde_prediction_ml_2024,
  title={BDE Prediction ML: Machine Learning for C-X Bond Dissociation Energy Prediction},
  author={Authors},
  year={2024},
  url={https://github.com/Dani-Sh113/bde-prediction-ml}
}
```

## Authors

- Original authors of the BDE prediction study

## License

MIT License - see LICENSE file for details

## Support & Contributing

For issues, questions, or contributions:
1. Open an issue on GitHub
2. Check existing documentation in `data/README.md` and `models/README.md`
3. Review commented code in `src/` modules

## Future Improvements

- [ ] Ensemble methods combining multiple models
- [ ] Hyperparameter optimization/tuning
- [ ] Uncertainty quantification in predictions
- [ ] GUI for easy prediction interface
- [ ] Molecular structure visualization
- [ ] Model explanation API

---

**Last Updated:** 2024
**Status:** ✅ Production Ready
