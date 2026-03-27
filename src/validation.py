"""
External dataset validation module.

Validates trained model performance on unseen external data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
from typing import Tuple

from . import config
from . import utils


# ============================================================================
# VALIDATION METRICS & ANALYSIS
# ============================================================================

def calculate_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive validation metrics for external data.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of validation metrics
    """
    metrics = utils.calculate_regression_metrics(y_true, y_pred)

    residuals = y_true - y_pred
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Statistical tests
    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    shapiro_stat, shapiro_p = stats.shapiro(residuals)

    metrics.update({
        'MAPE (%)': mape,
        'Pearson r': pearson_r,
        'Pearson p-value': pearson_p,
        'Spearman ρ': spearman_r,
        'Spearman p-value': spearman_p,
        'Shapiro-Wilk stat': shapiro_stat,
        'Shapiro-Wilk p-value': shapiro_p,
    })

    return metrics


def print_validation_results(metrics: dict, sample_count: int):
    """Print validation metrics in formatted table."""
    utils.print_section("EXTERNAL VALIDATION METRICS")

    print(f"\nSample Size: {sample_count}\n")

    print("Regression Metrics:")
    print(f"  R² Score:    {metrics['R² Score']:.4f}")
    print(f"  MAE:         {metrics['MAE (kJ/mol)']:.4f} kJ/mol")
    print(f"  RMSE:        {metrics['RMSE (kJ/mol)']:.4f} kJ/mol")
    print(f"  MSE:         {metrics['MSE']:.4f}")
    print(f"  MAPE:        {metrics['MAPE (%)']:.2f}%")

    print("\nCorrelation Metrics:")
    print(f"  Pearson r:   {metrics['Pearson r']:.4f} (p={metrics['Pearson p-value']:.4e})")
    print(f"  Spearman ρ:  {metrics['Spearman ρ']:.4f} (p={metrics['Spearman p-value']:.4e})")

    print("\nResidual Statistics:")
    print(f"  Mean:        {metrics['Mean Residual']:.4f} kJ/mol")
    print(f"  Std:         {metrics['Std Residual']:.4f} kJ/mol")

    print("\nNormality Test (Shapiro-Wilk):")
    print(f"  Statistic:   {metrics['Shapiro-Wilk stat']:.4f}")
    print(f"  p-value:     {metrics['Shapiro-Wilk p-value']:.4f}")
    if metrics['Shapiro-Wilk p-value'] > 0.05:
        print("  ✓ Residuals are normally distributed")
    else:
        print("  ⚠ Residuals may not be normally distributed")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_validation_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "validation_parity_plot"
):
    """Plot parity plot for validation set."""
    metrics = utils.calculate_regression_metrics(y_true, y_pred)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(10, 10))

    scatter = ax.scatter(
        y_true, y_pred,
        c=np.abs(residuals), cmap='viridis',
        alpha=0.7, s=100, edgecolors='black', linewidth=0.8
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')

    ax.fill_between(
        [min_val, max_val],
        [min_val - residuals.std(), max_val - residuals.std()],
        [min_val + residuals.std(), max_val + residuals.std()],
        alpha=0.15, color='red', label=f'±1 Std ({residuals.std():.2f} kJ/mol)'
    )

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Absolute Error (kJ/mol)', fontweight='bold')

    textstr = f'R² = {metrics["R² Score"]:.4f}\nMAE = {metrics["MAE (kJ/mol)"]:.2f}\nRMSE = {metrics["RMSE (kJ/mol)"]:.2f}\nn = {len(y_true)}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5))

    ax.set_xlabel('Actual BDE (kJ/mol)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Predicted BDE (kJ/mol)', fontweight='bold', fontsize=13)
    ax.set_title('Validation: Predicted vs Actual BDE', fontweight='bold', fontsize=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='validation')


def plot_validation_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "validation_residuals"
):
    """Plot residuals vs predictions for validation set."""
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(y_pred, residuals, alpha=0.6, s=80, edgecolors='black', linewidth=0.7, color='steelblue')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, label='Zero Error')
    ax.axhline(y=residuals.std(), color='orange', linestyle=':', linewidth=2,
              label=f'+1 Std ({residuals.std():.2f} kJ/mol)')
    ax.axhline(y=-residuals.std(), color='orange', linestyle=':', linewidth=2,
              label=f'-1 Std ({residuals.std():.2f} kJ/mol)')

    ax.fill_between(y_pred, residuals.std(), -residuals.std(), alpha=0.1, color='orange')

    ax.set_xlabel('Predicted BDE (kJ/mol)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Residual (kJ/mol)', fontweight='bold', fontsize=13)
    ax.set_title('Validation: Residual Analysis', fontweight='bold', fontsize=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='validation')


def plot_validation_multiplot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "validation_comprehensive"
):
    """Create comprehensive 4-panel validation analysis."""
    residuals = y_true - y_pred
    metrics = utils.calculate_regression_metrics(y_true, y_pred)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Residual distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='skyblue', linewidth=1.5)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2.5)
    ax1.axvline(x=residuals.mean(), color='green', linestyle='-', linewidth=2.5,
               label=f'Mean ({residuals.mean():.2f})')
    ax1.set_xlabel('Residual (kJ/mol)', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Residual Distribution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Panel 2: Q-Q Plot
    ax2 = fig.add_subplot(gs[0, 1])
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Absolute error vs actual
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_true, np.abs(residuals), alpha=0.6, s=80, edgecolors='black', linewidth=0.7, color='coral')
    ax3.axhline(y=metrics['MAE (kJ/mol)'], color='red', linestyle='--', linewidth=2.5,
               label=f'MAE ({metrics["MAE (kJ/mol)"]:.2f})')
    ax3.set_xlabel('Actual BDE (kJ/mol)', fontweight='bold')
    ax3.set_ylabel('Absolute Error (kJ/mol)', fontweight='bold')
    ax3.set_title('Error vs Actual Value', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Sample-by-sample comparison
    ax4 = fig.add_subplot(gs[1, 1])
    x_indices = np.arange(len(y_true))
    ax4.scatter(x_indices, y_true, marker='o', label='Actual', color='blue', s=70, zorder=5)
    ax4.scatter(x_indices, y_pred, marker='^', label='Predicted', color='red', s=70, zorder=5)
    for i, (actual, pred) in enumerate(zip(y_true, y_pred)):
        ax4.plot([i, i], [actual, pred], 'k-', alpha=0.2, linewidth=0.8)
    ax4.set_xlabel('Sample Index', fontweight='bold')
    ax4.set_ylabel('BDE (kJ/mol)', fontweight='bold')
    ax4.set_title('Sample-by-Sample Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('External Validation: Comprehensive Analysis', fontweight='bold', fontsize=16, y=0.995)
    plt.tight_layout()
    utils.save_figure(fig, filename, phase='validation')


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_validation_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    filename: str = "validation_summary"
):
    """Save validation results to CSV."""
    df_summary = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()]
    })

    utils.save_metrics_csv(df_summary.set_index('Metric').to_dict('index'), filename, phase='validation')


def save_validation_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "validation_predictions"
):
    """Save detailed prediction results to CSV."""
    residuals = y_true - y_pred

    df_results = pd.DataFrame({
        'Actual_BDE_kJ': y_true,
        'Predicted_BDE_kJ': y_pred,
        'Residual': residuals,
        'Abs_Residual': np.abs(residuals),
        'Percent_Error': (residuals / y_true) * 100
    })

    config.RESULTS_VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    filepath = config.RESULTS_VALIDATION_DIR / f"{filename}.csv"
    df_results.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


# ============================================================================
# MAIN VALIDATION FUNCTION
# ============================================================================

def run_external_validation(
    model,
    X_external: np.ndarray,
    y_external: np.ndarray,
    feature_names: list = None
):
    """
    Complete external validation pipeline.

    Args:
        model: Trained model
        X_external: External feature data
        y_external: External target data
        feature_names: Optional list of feature names
    """
    utils.print_section("EXTERNAL DATA VALIDATION")

    # Make predictions
    print("Generating predictions on external data...")
    y_pred = model.predict(X_external)
    print(f"✓ Predictions generated for {len(y_external)} samples")

    # Calculate metrics
    print("\nCalculating validation metrics...")
    metrics = calculate_validation_metrics(y_external, y_pred)
    print_validation_results(metrics, len(y_external))

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_validation_parity(y_external, y_pred)
    plot_validation_residuals(y_external, y_pred)
    plot_validation_multiplot(y_external, y_pred)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    save_validation_results(y_external, y_pred, metrics)
    save_validation_predictions(y_external, y_pred)

    print("\n✓ External validation complete!")

    return metrics
