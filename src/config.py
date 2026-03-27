"""
Configuration module for BDE Prediction ML project.

Contains all constants, file paths, feature definitions, and hyperparameters
used throughout the project. This is the single source of truth for configuration.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INTERIM_DIR = DATA_DIR / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, DATA_PROCESSED_DIR, DATA_INTERIM_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA FILE PATHS
# ============================================================================

FULL_DATASET_PATH = DATA_PROCESSED_DIR / "full_dataset_14_descriptors.csv"
UNSEEN_DATA_PATH = DATA_PROCESSED_DIR / "unseen_data.csv"
CORRELATIONS_PATH = DATA_INTERIM_DIR / "highly_correlated_pairs.csv"
DESCRIPTOR_LIST_PATH = DATA_INTERIM_DIR / "final_descriptor_list.csv"
TRAINED_MODEL_PATH = MODELS_DIR / "tpot_best_pipeline.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# ============================================================================
# MOLECULAR DESCRIPTOR FEATURES
# ============================================================================

# All 14 descriptors used in the project
ALL_DESCRIPTORS = [
    'MolWt',               # Molecular weight
    'LogP',                # Lipophilicity (hydrophobicity)
    'TPSA',                # Topological Polar Surface Area
    'NumRotBonds',         # Number of rotatable bonds
    'HBD',                 # Hydrogen bond donors
    'FractionCsp3',        # Fraction of sp3-hybridized carbons
    'NumHeavyAtoms',       # Number of non-hydrogen atoms
    'Halogen_Z',           # Atomic number of halogen (9=F, 17=Cl, 35=Br, 53=I)
    'C_degree',            # Degree of carbon bonded to halogen
    'C_hybridization',     # Hybridization of carbon (2=sp, 3=sp2, 4=sp3)
    'X_hybridization',     # Hybridization of halogen
    'C_GasteigerCharge',   # Gasteiger partial charge on carbon
    'X_GasteigerCharge',   # Gasteiger partial charge on halogen
    'X_EN',                # Electronegativity of halogen
]

# Non-feature columns to drop when preparing data
NON_FEATURE_COLS = [
    'Atoms',               # Compound name
    'BDE_kcal',            # Target in kcal/mol
    'BDE_kJ',              # Target in kJ/mol
    'SMILES',              # Chemical structure notation
    'Targeted_Halogen',    # Which halogen (F, Cl, Br, I)
    'Unnamed: 0'           # Index column from CSV
]

TARGET_VARIABLE = 'BDE_kJ'  # Primary target variable (kilojoules/mol)

# ============================================================================
# CORRELATION ANALYSIS SETTINGS
# ============================================================================

CORRELATION_THRESHOLD = 0.9  # Threshold for identifying highly correlated features
CORRELATION_METHOD = 'pearson'

# ============================================================================
# TPOT MODEL HYPERPARAMETERS
# ============================================================================

TPOT_GENERATIONS = 10
TPOT_POPULATION_SIZE = 50
TPOT_CV_FOLDS = 10
TPOT_RANDOM_STATE = 42
TPOT_N_JOBS = 1
TPOT_VERBOSE = 2

# ============================================================================
# DATA PREPROCESSING SETTINGS
# ============================================================================

TEST_SIZE = 0.2  # Fraction of data to use for testing
RANDOM_STATE = 42
IMPUTATION_STRATEGY = 'mean'  # Strategy for imputing missing values

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

DPI_RESOLUTION = 300  # Resolution for saved plots
FIGURE_FORMAT = ['png', 'pdf']  # Formats to save figures as
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 11

# Colormap for heatmaps
HEATMAP_CMAP = (250, 10)  # seaborn.diverging_palette parameters (hue_start, hue_end)

# ============================================================================
# SHAP ANALYSIS SETTINGS
# ============================================================================

SHAP_BACKGROUND_SAMPLES = 100  # Number of samples for SHAP background data
FEATURE_IMPORTANCE_THRESHOLD_PCT = 1.0  # Feature contribution threshold (%)
SHAP_SUMMARY_PLOT_SAMPLES = 20  # Samples for SHAP force plots

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

SHAPIRO_TEST_ALPHA = 0.05  # Significance level for Shapiro-Wilk test

# ============================================================================
# RESULT OUTPUT DIRECTORIES
# ============================================================================

RESULTS_CORRELATION_DIR = RESULTS_DIR / "correlation"
RESULTS_TRAINING_DIR = RESULTS_DIR / "training"
RESULTS_SHAP_DIR = RESULTS_DIR / "shap"
RESULTS_VALIDATION_DIR = RESULTS_DIR / "validation"

# Create subdirectories
for dir_path in [RESULTS_CORRELATION_DIR, RESULTS_TRAINING_DIR, RESULTS_SHAP_DIR, RESULTS_VALIDATION_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# UTILITY FUNCTIONS FOR CONFIG
# ============================================================================

def get_expected_features():
    """Get list of expected features after correlation analysis."""
    return ALL_DESCRIPTORS


def get_result_dir(phase_name):
    """Get result directory for a specific analysis phase."""
    phase_dirs = {
        'correlation': RESULTS_CORRELATION_DIR,
        'training': RESULTS_TRAINING_DIR,
        'shap': RESULTS_SHAP_DIR,
        'validation': RESULTS_VALIDATION_DIR,
    }
    return phase_dirs.get(phase_name, RESULTS_DIR)
