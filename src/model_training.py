"""
Model training module using TPOT (Tree-based Pipeline Optimization Tool).

Handles:
- TPOT model training and optimization
- Model evaluation on test set
- Performance visualization
- Model persistence
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tpot import TPOTRegressor
from typing import Tuple, Any, Dict

from . import config
from . import utils


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_tpot_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    generations: int = config.TPOT_GENERATIONS,
    population_size: int = config.TPOT_POPULATION_SIZE,
    cv_folds: int = config.TPOT_CV_FOLDS,
    random_state: int = config.TPOT_RANDOM_STATE
) -> TPOTRegressor:
    """
    Train TPOT regressor model.

    Args:
        X_train: Training features
        y_train: Training target
        generations: Number of generations (iterations)
        population_size: Population size per generation
        cv_folds: K-fold cross-validation folds
        random_state: Random seed

    Returns:
        Fitted TPOT model
    """
    utils.print_section("TPOT MODEL TRAINING")

    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Hyperparameters:")
    print(f"  Generations: {generations}")
    print(f"  Population size: {population_size}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Random state: {random_state}\n")

    tpot = TPOTRegressor(
        generations=generations,
        population_size=population_size,
        cv=cv_folds,
        verbose=config.TPOT_VERBOSE,
        random_state=random_state,
        n_jobs=config.TPOT_N_JOBS
    )

    print("Starting TPOT optimization...")
    print("(This may take several minutes or hours depending on hyperparameters)\n")

    start_time = time.time()
    tpot.fit(X_train, y_train)
    end_time = time.time()

    total_time = end_time - start_time
    print(f"\n✓ TPOT training complete!")
    print(f"  Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    return tpot


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on training and test sets.

    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Dictionary with metrics for train and test sets
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    # Training set predictions
    y_train_pred = model.predict(X_train)
    train_metrics = utils.calculate_regression_metrics(y_train, y_train_pred)

    # Test set predictions
    y_test_pred = model.predict(X_test)
    test_metrics = utils.calculate_regression_metrics(y_test, y_test_pred)

    # Print results
    print("\nTraining Set:")
    for key, value in train_metrics.items():
        print(f"  {key:.<35} {value:>10.4f}")

    print("\nTest Set:")
    for key, value in test_metrics.items():
        print(f"  {key:.<35} {value:>10.4f}")

    results = {
        'train': train_metrics,
        'test': test_metrics,
        'train_predictions': y_train_pred,
        'test_predictions': y_test_pred,
    }

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_parity_plots(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    filename: str = "parity_plots"
):
    """Create side-by-side parity plots for training and test sets."""
    train_metrics = utils.calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = utils.calculate_regression_metrics(y_test, y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Training set
    ax = axes[0]
    ax.scatter(y_train, y_train_pred, alpha=0.6, s=60, edgecolors='black',
               linewidth=0.5, color='blue')
    min_val = min(y_train.min(), y_train_pred.min())
    max_val = max(y_train.max(), y_train_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    textstr = f"R² = {train_metrics['R² Score']:.4f}\nMAE = {train_metrics['MAE (kJ/mol)']:.2f}\nRMSE = {train_metrics['RMSE (kJ/mol)']:.2f}\nn = {len(y_train)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax.set_xlabel('Actual BDE (kJ/mol)', fontweight='bold')
    ax.set_ylabel('Predicted BDE (kJ/mol)', fontweight='bold')
    ax.set_title('Training Set', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Test set
    ax = axes[1]
    ax.scatter(y_test, y_test_pred, alpha=0.6, s=60, edgecolors='black',
               linewidth=0.5, color='green')
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    textstr = f"R² = {test_metrics['R² Score']:.4f}\nMAE = {test_metrics['MAE (kJ/mol)']:.2f}\nRMSE = {test_metrics['RMSE (kJ/mol)']:.2f}\nn = {len(y_test)}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.set_xlabel('Actual BDE (kJ/mol)', fontweight='bold')
    ax.set_ylabel('Predicted BDE (kJ/mol)', fontweight='bold')
    ax.set_title('Test Set', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='training')


def plot_combined_parity(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    filename: str = "combined_parity_plot"
):
    """Create combined parity plot with both train and test data."""
    train_metrics = utils.calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = utils.calculate_regression_metrics(y_test, y_test_pred)

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(y_train, y_train_pred, alpha=0.5, s=60, label=f'Training (n={len(y_train)})',
               color='blue')
    ax.scatter(y_test, y_test_pred, alpha=0.7, s=80, label=f'Test (n={len(y_test)})',
               color='red', marker='^')

    min_val = min(y_train.min(), y_test.min(), y_train_pred.min(), y_test_pred.min())
    max_val = max(y_train.max(), y_test.max(), y_train_pred.max(), y_test_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)

    textstr = f"Training: R²={train_metrics['R² Score']:.4f}, RMSE={train_metrics['RMSE (kJ/mol)']:.2f}\nTest: R²={test_metrics['R² Score']:.4f}, RMSE={test_metrics['RMSE (kJ/mol)']:.2f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Actual BDE (kJ/mol)', fontweight='bold', fontsize=14)
    ax.set_ylabel('Predicted BDE (kJ/mol)', fontweight='bold', fontsize=14)
    ax.set_title('Training & Test Set Performance', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='training')


def plot_residuals(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    filename: str = "residual_analysis"
):
    """Create 4-panel residual analysis plot."""
    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Training residuals vs predictions
    axes[0, 0].scatter(y_train_pred, train_residuals, alpha=0.6, s=60, color='blue')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted BDE (kJ/mol)', fontweight='bold')
    axes[0, 0].set_ylabel('Residual (kJ/mol)', fontweight='bold')
    axes[0, 0].set_title('Training: Residual Plot', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Test residuals vs predictions
    axes[0, 1].scatter(y_test_pred, test_residuals, alpha=0.6, s=60, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted BDE (kJ/mol)', fontweight='bold')
    axes[0, 1].set_ylabel('Residual (kJ/mol)', fontweight='bold')
    axes[0, 1].set_title('Test: Residual Plot', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Training residual distribution
    axes[1, 0].hist(train_residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residual (kJ/mol)', fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontweight='bold')
    axes[1, 0].set_title('Training: Residual Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Test residual distribution
    axes[1, 1].hist(test_residuals, bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Residual (kJ/mol)', fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontweight='bold')
    axes[1, 1].set_title('Test: Residual Distribution', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='training')


def plot_metrics_comparison(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    filename: str = "metrics_comparison"
):
    """Create bar chart comparing training vs test metrics."""
    train_metrics = utils.calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = utils.calculate_regression_metrics(y_test, y_test_pred)

    metrics_names = ['R² Score', 'MAE (kJ/mol)', 'RMSE (kJ/mol)']
    train_vals = [
        train_metrics['R² Score'],
        train_metrics['MAE (kJ/mol)'],
        train_metrics['RMSE (kJ/mol)']
    ]
    test_vals = [
        test_metrics['R² Score'],
        test_metrics['MAE (kJ/mol)'],
        test_metrics['RMSE (kJ/mol)']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_vals, width, label='Training', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, test_vals, width, label='Test', color='lightcoral', edgecolor='black')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Value', fontweight='bold', fontsize=12)
    ax.set_title('Training vs Test Performance', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    utils.save_figure(fig, filename, phase='training')


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_metrics_to_csv(
    y_train: np.ndarray,
    y_train_pred: np.ndarray,
    y_test: np.ndarray,
    y_test_pred: np.ndarray,
    training_time: float,
    filename: str = "training_metrics"
):
    """Save training and test metrics to CSV."""
    train_metrics = utils.calculate_regression_metrics(y_train, y_train_pred)
    test_metrics = utils.calculate_regression_metrics(y_test, y_test_pred)

    df_metrics = pd.DataFrame({
        'Metric': ['R² Score', 'MAE (kJ/mol)', 'RMSE (kJ/mol)', 'MSE', 'Mean Residual', 'Std Residual'],
        'Training': [
            f"{train_metrics['R² Score']:.4f}",
            f"{train_metrics['MAE (kJ/mol)']:.4f}",
            f"{train_metrics['RMSE (kJ/mol)']:.4f}",
            f"{train_metrics['MSE']:.4f}",
            f"{train_metrics['Mean Residual']:.4f}",
            f"{train_metrics['Std Residual']:.4f}"
        ],
        'Test': [
            f"{test_metrics['R² Score']:.4f}",
            f"{test_metrics['MAE (kJ/mol)']:.4f}",
            f"{test_metrics['RMSE (kJ/mol)']:.4f}",
            f"{test_metrics['MSE']:.4f}",
            f"{test_metrics['Mean Residual']:.4f}",
            f"{test_metrics['Std Residual']:.4f}"
        ]
    })

    utils.save_metrics_csv(df_metrics.set_index('Metric').to_dict('index'), filename, phase='training')


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run_model_training(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Any:
    """
    Complete training pipeline: train model, evaluate, visualize, save.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        Trained TPOT model
    """
    # Train model
    model = train_tpot_model(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_train, X_test, y_train, y_test)

    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_parity_plots(y_train, results['train_predictions'], y_test, results['test_predictions'])
    plot_combined_parity(y_train, results['train_predictions'], y_test, results['test_predictions'])
    plot_residuals(y_train, results['train_predictions'], y_test, results['test_predictions'])
    plot_metrics_comparison(y_train, results['train_predictions'], y_test, results['test_predictions'])

    # Save model and metrics
    utils.save_model(model)
    save_metrics_to_csv(y_train, results['train_predictions'], y_test, results['test_predictions'], 0)

    print("\n✓ Model training complete!")
    return model
