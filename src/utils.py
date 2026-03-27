"""
Utility functions for BDE Prediction ML project.

Contains shared functions for:
- Data visualization
- Model loading/saving
- Metrics calculation
- File operations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path
from typing import Tuple, Optional, List

from . import config

warnings.filterwarnings('ignore')


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def set_plot_style():
    """Set consistent matplotlib and seaborn styling for plots."""
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = config.FONT_SIZE_TICK
    plt.rcParams['axes.labelsize'] = config.FONT_SIZE_LABEL
    plt.rcParams['axes.titlesize'] = config.FONT_SIZE_TITLE
    plt.rcParams['xtick.labelsize'] = config.FONT_SIZE_TICK
    plt.rcParams['ytick.labelsize'] = config.FONT_SIZE_TICK
    plt.rcParams['legend.fontsize'] = config.FONT_SIZE_TICK
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.5


def save_figure(fig, filename: str, phase: str = 'results', formats: List[str] = None):
    """
    Save a matplotlib figure to disk in multiple formats.

    Args:
        fig: matplotlib figure object
        filename: Name of file (without extension)
        phase: Phase directory ('correlation', 'training', 'shap', 'validation')
        formats: List of formats to save (['png', 'pdf'] by default)
    """
    if formats is None:
        formats = config.FIGURE_FORMAT

    result_dir = config.get_result_dir(phase)
    result_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = result_dir / f"{filename}.{fmt}"
        fig.savefig(
            filepath,
            dpi=config.DPI_RESOLUTION,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        print(f"✓ Saved: {filepath.name}")

    plt.close(fig)


def create_heatmap(corr_matrix: pd.DataFrame, title: str, figsize: Optional[Tuple] = None) -> plt.Figure:
    """
    Create a styled correlation heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame
        title: Title for the heatmap
        figsize: Figure size (width, height)

    Returns:
        matplotlib figure object
    """
    if figsize is None:
        n_features = len(corr_matrix.columns)
        figsize = (max(12, n_features * 1.2), max(10, n_features * 1.2))

    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    cmap = sns.diverging_palette(
        config.HEATMAP_CMAP[0],
        config.HEATMAP_CMAP[1],
        s=95, l=45, as_cmap=True
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        square=True,
        linewidths=1.0,
        linecolor='white',
        cbar_kws={'shrink': 0.8, 'aspect': 20},
        vmin=-1,
        vmax=1,
        annot_kws={'size': 9, 'weight': 'bold'},
        ax=ax
    )

    ax.set_title(title, fontsize=config.FONT_SIZE_TITLE, weight='bold', pad=20)
    plt.tight_layout()

    return fig


# ============================================================================
# MODEL UTILITIES
# ============================================================================

def save_model(model, filepath: Path = None):
    """
    Save trained model using joblib.

    Args:
        model: Trained model object
        filepath: Path to save model (defaults to config.TRAINED_MODEL_PATH)
    """
    if filepath is None:
        filepath = config.TRAINED_MODEL_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(filepath))
    print(f"✓ Model saved to: {filepath}")


def load_model(filepath: Path = None):
    """
    Load trained model from disk.

    Args:
        filepath: Path to model file (defaults to config.TRAINED_MODEL_PATH)

    Returns:
        Loaded model object
    """
    if filepath is None:
        filepath = config.TRAINED_MODEL_PATH

    if not filepath.exists():
        raise FileNotFoundError(f"Model not found at: {filepath}")

    model = joblib.load(str(filepath))
    print(f"✓ Model loaded from: {filepath}")
    return model


def save_scaler(scaler, filepath: Path = None):
    """Save data scaler/imputer using joblib."""
    if filepath is None:
        filepath = config.SCALER_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(filepath))
    print(f"✓ Scaler saved to: {filepath}")


def load_scaler(filepath: Path = None):
    """Load data scaler/imputer from disk."""
    if filepath is None:
        filepath = config.SCALER_PATH

    if not filepath.exists():
        raise FileNotFoundError(f"Scaler not found at: {filepath}")

    scaler = joblib.load(str(filepath))
    print(f"✓ Scaler loaded from: {filepath}")
    return scaler


# ============================================================================
# METRICS & STATISTICS
# ============================================================================

def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    residuals = y_true - y_pred
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    metrics = {
        'R² Score': r2,
        'MAE (kJ/mol)': mae,
        'RMSE (kJ/mol)': rmse,
        'MSE': mse,
        'MAPE (%)': mape,
        'Mean Residual': mean_residual,
        'Std Residual': std_residual,
    }

    return metrics


def print_metrics_table(metrics: dict, label: str = "Metrics"):
    """Print metrics in a formatted table."""
    print("\n" + "="*60)
    print(label)
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:.<40} {value:>15.4f}")
        else:
            print(f"{key:.<40} {value:>15}")
    print("="*60)


def save_metrics_csv(metrics: dict, filename: str, phase: str = 'results'):
    """Save metrics to CSV file."""
    result_dir = config.get_result_dir(phase)
    result_dir.mkdir(parents=True, exist_ok=True)

    filepath = result_dir / f"{filename}.csv"
    df_metrics = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
    df_metrics.to_csv(filepath, index=False)
    print(f"✓ Metrics saved to: {filepath.name}")


# ============================================================================
# DATA UTILITIES
# ============================================================================

def ensure_columns_exist(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """Check if all required columns exist in dataframe."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"✗ ERROR: Missing columns: {missing}")
        return False
    return True


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Remove non-feature columns from dataframe."""
    cols_to_drop = [col for col in config.NON_FEATURE_COLS if col in df.columns]
    return df.drop(columns=cols_to_drop, errors='ignore')


# ============================================================================
# FILE I/O
# ============================================================================

def ensure_directory_exists(directory: Path):
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def log_file_created(filepath: Path):
    """Log that a file was created."""
    print(f"✓ Saved: {filepath.name}")
