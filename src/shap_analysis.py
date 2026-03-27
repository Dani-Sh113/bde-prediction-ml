"""
SHAP (SHapley Additive exPlanations) Feature Importance Analysis module.

Analyzes feature contributions to model predictions using SHAP values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import warnings
from typing import Tuple

from . import config
from . import utils

warnings.filterwarnings('ignore')


# ============================================================================
# SHAP INITIALIZATION & CALCULATION
# ============================================================================

def initialize_shap_explainer(model, X_background: np.ndarray):
    """
    Initialize SHAP explainer for the model.

    Args:
        model: Trained model
        X_background: Background data for SHAP

    Returns:
        SHAP Explainer object
    """
    print("\n" + "="*70)
    print("INITIALIZING SHAP EXPLAINER")
    print("="*70)

    try:
        explainer = shap.TreeExplainer(model, X_background)
        print("✓ Using TreeExplainer (optimized for tree-based models)")
        return explainer
    except Exception as e:
        print(f"⚠ TreeExplainer failed, using KernelExplainer: {e}")
        explainer = shap.KernelExplainer(model.predict, X_background)
        print("✓ Using KernelExplainer")
        return explainer


def calculate_shap_values(explainer, X_test: pd.DataFrame) -> np.ndarray:
    """
    Calculate SHAP values for test set.

    Args:
        explainer: SHAP Explainer
        X_test: Test features

    Returns:
        SHAP values array
    """
    print("\n" + "="*70)
    print("CALCULATING SHAP VALUES")
    print("="*70)
    print("This may take a few minutes...\n")

    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, shap.Explanation):
        shap_values_array = shap_values.values
    else:
        shap_values_array = shap_values

    print(f"✓ SHAP values calculated for {X_test.shape[0]} samples")
    return shap_values_array


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def analyze_feature_importance(
    shap_values: np.ndarray,
    feature_names: list
) -> pd.DataFrame:
    """
    Analyze feature importance from SHAP values.

    Args:
        shap_values: SHAP values array
        feature_names: List of feature names

    Returns:
        DataFrame with feature importance metrics
    """
    print("\n" + "="*70)
    print("ANALYZING FEATURE CONTRIBUTIONS")
    print("="*70)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
        'Mean_SHAP': shap_values.mean(axis=0),
        'Std_SHAP': shap_values.std(axis=0),
        'Max_SHAP': np.abs(shap_values).max(axis=0)
    })

    feature_importance = feature_importance.sort_values('Mean_Abs_SHAP', ascending=False)
    feature_importance['Rank'] = range(1, len(feature_importance) + 1)
    feature_importance['Contribution_%'] = (
        feature_importance['Mean_Abs_SHAP'] / feature_importance['Mean_Abs_SHAP'].sum()
    ) * 100

    print("\nFeature Importance Ranking:")
    print(feature_importance[['Rank', 'Feature', 'Mean_Abs_SHAP', 'Contribution_%']].to_string(index=False))

    return feature_importance


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_shap_summary_bar(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    filename: str = "shap_feature_importance_bar"
):
    """Plot SHAP feature importance bar plot."""
    print("\nGenerating SHAP summary plot (bar)...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, ax=ax)
    ax.set_title("SHAP Feature Importance", fontweight='bold', fontsize=14)

    utils.save_figure(fig, filename, phase='shap')


def plot_shap_summary_beeswarm(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    filename: str = "shap_summary_beeswarm"
):
    """Plot SHAP beeswarm summary plot."""
    print("Generating SHAP summary plot (beeswarm)...")

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    ax.set_title("SHAP Summary Plot", fontweight='bold', fontsize=14)

    utils.save_figure(fig, filename, phase='shap')


def plot_shap_dependence_top_features(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list,
    n_features: int = 3
):
    """Plot SHAP dependence plots for top N features."""
    print(f"\nGenerating SHAP dependence plots for top {n_features} features...")

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-n_features:][::-1]

    for feature_idx in top_indices:
        feature_name = feature_names[feature_idx]
        print(f"  Generating plot for {feature_name}...")

        fig, ax = plt.subplots(figsize=(8, 6))
        shap.dependence_plot(
            feature_idx, shap_values, X_test,
            feature_names=feature_names, show=False, ax=ax
        )
        ax.set_title(f"SHAP Dependence: {feature_name}", fontweight='bold', fontsize=14)

        utils.save_figure(fig, f"shap_dependence_{feature_name}", phase='shap')


def plot_shap_contribution_pie(
    feature_importance: pd.DataFrame,
    n_top_features: int = 10,
    filename: str = "shap_contribution_pie_chart"
):
    """Plot feature contribution distribution as pie chart."""
    print("Generating SHAP contribution pie chart...")

    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = feature_importance.head(n_top_features)
    other_contrib = feature_importance.iloc[n_top_features:]['Contribution_%'].sum()

    pie_data = list(top_features['Contribution_%'].values)
    pie_labels = list(top_features['Feature'].values)

    if other_contrib > 0:
        pie_data.append(other_contrib)
        pie_labels.append(f'Others ({len(feature_importance)-n_top_features})')

    colors = plt.cm.Set3(range(len(pie_data)))
    wedges, texts, autotexts = ax.pie(
        pie_data, labels=pie_labels, autopct='%1.1f%%',
        startangle=90, colors=colors
    )

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    ax.set_title('Feature Contribution Distribution', fontweight='bold', fontsize=14)
    plt.tight_layout()

    utils.save_figure(fig, filename, phase='shap')


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_feature_importance_csv(
    feature_importance: pd.DataFrame,
    filename: str = "shap_feature_importance_analysis"
):
    """Save feature importance analysis to CSV."""
    config.RESULTS_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    filepath = config.RESULTS_SHAP_DIR / f"{filename}.csv"

    feature_importance.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


def save_shap_values_csv(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    model_predictions: np.ndarray,
    feature_names: list,
    filename: str = "shap_values_detailed"
):
    """Save detailed SHAP values to CSV."""
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['Actual_BDE_kJ'] = y_test
    shap_df['Predicted_BDE_kJ'] = model_predictions

    config.RESULTS_SHAP_DIR.mkdir(parents=True, exist_ok=True)
    filepath = config.RESULTS_SHAP_DIR / f"{filename}.csv"

    shap_df.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_shap_analysis(
    model,
    X_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    feature_names: list
):
    """
    Complete SHAP analysis pipeline.

    Args:
        model: Trained model
        X_train: Training features (for background data)
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
    """
    utils.print_section("SHAP FEATURE IMPORTANCE ANALYSIS")

    # Sample background data
    print("Sampling background data...")
    background_indices = np.random.choice(
        X_train.shape[0],
        min(config.SHAP_BACKGROUND_SAMPLES, X_train.shape[0]),
        replace=False
    )
    X_background = X_train[background_indices]
    print(f"✓ Using {len(background_indices)} background samples")

    # Initialize SHAP
    explainer = initialize_shap_explainer(model, X_background)

    # Calculate SHAP values
    shap_values = calculate_shap_values(explainer, X_test)

    # Analyze feature importance
    feature_importance = analyze_feature_importance(shap_values, feature_names)

    # Generate visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    plot_shap_summary_bar(shap_values, X_test)
    plot_shap_summary_beeswarm(shap_values, X_test)
    plot_shap_dependence_top_features(shap_values, X_test, feature_names, n_features=3)
    plot_shap_contribution_pie(feature_importance)

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    save_feature_importance_csv(feature_importance)
    model_predictions = model.predict(X_test)
    save_shap_values_csv(shap_values, X_test, y_test, model_predictions, feature_names)

    print("\n✓ SHAP analysis complete!")

    return feature_importance
