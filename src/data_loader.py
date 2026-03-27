"""
Data loading and preprocessing module.

Handles:
- Loading CSV data files
- Data validation and cleaning
- Feature selection and preprocessing
- Train/test splitting
- Missing value imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional

from . import config
from . import utils


# ============================================================================
# DATA LOADING
# ============================================================================

def load_full_dataset(filepath: Path = None) -> pd.DataFrame:
    """
    Load the full dataset with all 14 descriptors.

    Args:
        filepath: Path to CSV file (defaults to config.FULL_DATASET_PATH)

    Returns:
        DataFrame with all data
    """
    if filepath is None:
        filepath = config.FULL_DATASET_PATH

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✓ Loaded dataset: {filepath.name}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")

    # Validate that we have the expected columns
    if config.TARGET_VARIABLE not in df.columns:
        raise ValueError(f"Target variable '{config.TARGET_VARIABLE}' not found in dataset")

    return df


def load_external_dataset(filepath: Path = None) -> pd.DataFrame:
    """
    Load external/unseen dataset for validation.

    Args:
        filepath: Path to CSV file (defaults to config.UNSEEN_DATA_PATH)

    Returns:
        DataFrame with external data
    """
    if filepath is None:
        filepath = config.UNSEEN_DATA_PATH

    if not filepath.exists():
        raise FileNotFoundError(f"External dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✓ Loaded external dataset: {filepath.name}")
    print(f"  Shape: {df.shape}")

    return df


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def prepare_features_and_target(df: pd.DataFrame, use_descriptors: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from dataset.

    Args:
        df: Input dataframe
        use_descriptors: List of specific descriptors to use (None = all)

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Extract target
    if config.TARGET_VARIABLE not in df.columns:
        raise ValueError(f"Target '{config.TARGET_VARIABLE}' not found")

    y = df[config.TARGET_VARIABLE]

    # Extract features
    X = df.drop(columns=config.NON_FEATURE_COLS, errors='ignore')

    # If specific descriptors requested, select only those
    if use_descriptors is not None:
        available_cols = [col for col in use_descriptors if col in X.columns]
        X = X[available_cols]
        print(f"  Using {len(available_cols)} selected descriptors")
    else:
        print(f"  Using all {len(X.columns)} descriptors")

    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")

    return X, y


def handle_missing_values(X: np.ndarray, strategy: str = 'mean') -> Tuple[np.ndarray, SimpleImputer]:
    """
    Handle missing values in feature matrix.

    Args:
        X: Feature matrix (numpy array or DataFrame)
        strategy: Imputation strategy ('mean', 'median', 'most_frequent')

    Returns:
        Tuple of (imputed array, fitted imputer object)
    """
    imputer = SimpleImputer(strategy=strategy)
    X_imputed = imputer.fit_transform(X)

    n_missing = np.isnan(X).sum()
    if n_missing > 0:
        print(f"  Imputed {n_missing} missing values using '{strategy}' strategy")
    else:
        print(f"  No missing values found")

    return X_imputed, imputer


# ============================================================================
# TRAIN/TEST SPLITTING
# ============================================================================

def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.

    Args:
        X: Feature matrix
        y: Target vector
        test_size: Fraction for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test


# ============================================================================
# INTEGRATED LOADING PIPELINES
# ============================================================================

def load_and_prepare_training_data(
    filepath: Path = None,
    use_descriptors: Optional[list] = None,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """
    Complete pipeline: load data, prepare features, handle missing values, split.

    Args:
        filepath: Path to training data CSV
        use_descriptors: Optional list of descriptors to use
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    utils.print_section("LOADING AND PREPARING DATA")

    # Load dataset
    df = load_full_dataset(filepath)

    # Prepare features and target
    print("\nExtracting features and target...")
    X, y = prepare_features_and_target(df, use_descriptors)
    feature_names = X.columns

    # Handle missing values
    print("\nHandling missing values...")
    X_imputed, imputer = handle_missing_values(X.values, strategy=config.IMPUTATION_STRATEGY)

    # Save imputer for later use
    utils.save_scaler(imputer)

    # Split data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_train_test(
        X_imputed, y.values,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, feature_names


def load_and_prepare_external_data(
    filepath: Path = None,
    use_descriptors: Optional[list] = None
) -> Tuple[np.ndarray, pd.Series, pd.Index]:
    """
    Load and prepare external validation dataset.

    Args:
        filepath: Path to external data CSV
        use_descriptors: Optional list of descriptors to use

    Returns:
        Tuple of (X_external, y_external, feature_names)
    """
    utils.print_section("LOADING EXTERNAL DATA")

    # Load external dataset
    df = load_external_dataset(filepath)

    # Prepare features and target
    print("\nExtracting features and target...")
    X, y = prepare_features_and_target(df, use_descriptors)
    feature_names = X.columns

    # Use saved imputer if available
    print("\nApplying imputation...")
    try:
        imputer = utils.load_scaler()
        X_imputed = imputer.transform(X.values)
        print("  ✓ Used saved imputer")
    except FileNotFoundError:
        X_imputed, imputer = handle_missing_values(X.values)
        print("  ✓ Created new imputer (no saved scaler found)")

    return X_imputed, y.values, feature_names


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_data(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Validate dataset integrity and completeness.

    Args:
        df: DataFrame to validate
        verbose: Print validation details

    Returns:
        True if data is valid, False otherwise
    """
    if verbose:
        print("\n" + "="*60)
        print("DATA VALIDATION")
        print("="*60)

    # Check shape
    if df.shape[0] == 0 or df.shape[1] == 0:
        if verbose:
            print("✗ ERROR: Empty dataframe")
        return False

    if verbose:
        print(f"✓ Shape: {df.shape}")

    # Check for required columns
    required_cols = config.ALL_DESCRIPTORS + [config.TARGET_VARIABLE]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        if verbose:
            print(f"✗ ERROR: Missing columns: {missing_cols}")
        return False

    if verbose:
        print(f"✓ All {len(config.ALL_DESCRIPTORS)} required descriptors present")

    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if verbose:
        if missing_count > 0:
            print(f"⚠ Warning: {missing_count} missing values found")
        else:
            print(f"✓ No missing values")

    # Check target variable range
    y_min = df[config.TARGET_VARIABLE].min()
    y_max = df[config.TARGET_VARIABLE].max()
    if verbose:
        print(f"✓ Target range: {y_min:.2f} - {y_max:.2f} kJ/mol")

    return True
