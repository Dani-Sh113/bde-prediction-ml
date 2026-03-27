"""
Feature correlation analysis module.

Analyzes feature correlations, identifies redundant features,
and generates visualizations and descriptor lists.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple

from . import config
from . import utils


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def calculate_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix for features.

    Args:
        df: DataFrame with features
        method: Correlation method ('pearson', 'spearman', 'kendall')

    Returns:
        Correlation matrix
    """
    print(f"Calculating {method} correlation matrix...")
    corr_matrix = df.corr(method=method)
    print(f"✓ Correlation matrix computed: {corr_matrix.shape}")
    return corr_matrix


def find_highly_correlated_features(
    corr_matrix: pd.DataFrame,
    threshold: float = config.CORRELATION_THRESHOLD
) -> pd.DataFrame:
    """
    Identify pairs of features with correlation above threshold.

    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (|r| > threshold)

    Returns:
        DataFrame of high-correlation pairs
    """
    corr_abs = corr_matrix.abs()
    high_corr_pairs = []

    for i in range(len(corr_abs.columns)):
        for j in range(i + 1, len(corr_abs.columns)):
            if corr_abs.iloc[i, j] > threshold:
                high_corr_pairs.append({
                    'Descriptor_1': corr_abs.columns[i],
                    'Descriptor_2': corr_abs.columns[j],
                    'Correlation': corr_matrix.iloc[i, j],
                    'Abs_Correlation': corr_abs.iloc[i, j]
                })

    if high_corr_pairs:
        df_pairs = pd.DataFrame(high_corr_pairs)
        df_pairs = df_pairs.sort_values('Abs_Correlation', ascending=False)
        print(f"✓ Found {len(df_pairs)} highly correlated pairs (threshold={threshold})")
        return df_pairs
    else:
        print(f"ℹ No highly correlated pairs found (threshold={threshold})")
        return pd.DataFrame()


def determine_descriptors_to_remove(
    high_corr_pairs: pd.DataFrame
) -> List[str]:
    """
    Determine which descriptors to remove based on correlation analysis.

    Strategy: From each correlated pair, remove the second descriptor
    (keeping order from the original feature list).

    Args:
        high_corr_pairs: DataFrame of high-correlation pairs

    Returns:
        List of descriptor names to remove
    """
    features_to_remove = []

    for _, row in high_corr_pairs.iterrows():
        desc2 = row['Descriptor_2']
        if desc2 not in features_to_remove:
            features_to_remove.append(desc2)

    print(f"\n🗑️  Features to remove:")
    for feature in features_to_remove:
        print(f"   - {feature}")

    return features_to_remove


def get_final_descriptors(
    all_descriptors: List[str],
    features_to_remove: List[str]
) -> List[str]:
    """
    Get final list of descriptors after removing redundant features.

    Args:
        all_descriptors: Original list of all descriptors
        features_to_remove: List of features to remove

    Returns:
        Final list of descriptor names
    """
    final_descriptors = [d for d in all_descriptors if d not in features_to_remove]
    print(f"\n Original descriptors: {len(all_descriptors)}")
    print(f"  Removed: {len(features_to_remove)}")
    print(f"  Final: {len(final_descriptors)}")
    return final_descriptors


# ============================================================================
# VISUALIZATION
# ============================================================================

def generate_correlation_heatmap(
    df: pd.DataFrame,
    title: str,
    output_filename: str,
    figsize: Tuple[int, int] = None
):
    """
    Generate and save correlation heatmap.

    Args:
        df: DataFrame with features to correlate
        title: Title for heatmap
        output_filename: Name for output file (without extension)
        figsize: Optional figure size
    """
    print(f"\nGenerating heatmap: {title}")

    corr_matrix = df.corr(method='pearson')

    fig = utils.create_heatmap(corr_matrix, title, figsize)
    utils.save_figure(fig, output_filename, phase='correlation')


def print_correlation_summary(
    original_count: int,
    high_corr_pairs: pd.DataFrame,
    removed_count: int,
    final_count: int
):
    """Print summary of correlation analysis."""
    utils.print_section("CORRELATION ANALYSIS SUMMARY")
    print(f"Original descriptors:      {original_count}")
    print(f"High-correlation pairs:    {len(high_corr_pairs)}")
    print(f"Descriptors removed:       {removed_count}")
    print(f"Final descriptors:         {final_count}")
    print(f"Reduction:                 {(1 - final_count/original_count)*100:.1f}%")


def save_high_corr_pairs(
    high_corr_pairs: pd.DataFrame,
    filepath: Path = None
):
    """Save high-correlation pairs to CSV."""
    if filepath is None:
        filepath = config.CORRELATIONS_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)
    high_corr_pairs.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


def save_final_descriptors(
    final_descriptors: List[str],
    filepath: Path = None
):
    """Save final descriptor list to CSV."""
    if filepath is None:
        filepath = config.DESCRIPTOR_LIST_PATH

    filepath.parent.mkdir(parents=True, exist_ok=True)

    df_descriptors = pd.DataFrame({
        'Descriptor': final_descriptors,
        'Status': 'Retained'
    })

    df_descriptors.to_csv(filepath, index=False)
    print(f"✓ Saved: {filepath.name}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_correlation_analysis(
    df: pd.DataFrame,
    threshold: float = config.CORRELATION_THRESHOLD,
    generate_plots: bool = True
) -> Tuple[List[str], pd.DataFrame]:
    """
    Complete correlation analysis pipeline.

    Args:
        df: DataFrame with features
        threshold: Correlation threshold for identifying redundant features
        generate_plots: Whether to generate visualization plots

    Returns:
        Tuple of (final_descriptors, high_corr_pairs)
    """
    utils.print_section("FEATURE CORRELATION ANALYSIS")

    # Extract only descriptor columns
    descriptor_cols = [col for col in config.ALL_DESCRIPTORS if col in df.columns]
    df_features = df[descriptor_cols]

    print(f"Analyzing {len(descriptor_cols)} descriptors...\n")

    # Calculate correlation
    corr_matrix = calculate_correlation_matrix(df_features)

    # Find high-correlation pairs
    high_corr_pairs = find_highly_correlated_features(corr_matrix, threshold)

    # Generate all-descriptor heatmap
    if generate_plots:
        print("\nGenerating visualizations...")
        generate_correlation_heatmap(
            df_features,
            f'Correlation Matrix - All {len(descriptor_cols)} Descriptors',
            'correlation_heatmap_all_descriptors'
        )

    # Determine which features to remove
    if len(high_corr_pairs) > 0:
        features_to_remove = determine_descriptors_to_remove(high_corr_pairs)
        final_descriptors = get_final_descriptors(descriptor_cols, features_to_remove)

        # Generate reduced heatmap
        if generate_plots:
            df_reduced = df[final_descriptors]
            generate_correlation_heatmap(
                df_reduced,
                f'Correlation Matrix - {len(final_descriptors)} Final Descriptors',
                'correlation_heatmap_reduced_descriptors'
            )
    else:
        final_descriptors = descriptor_cols
        features_to_remove = []

    # Print summary
    print_correlation_summary(
        len(descriptor_cols),
        high_corr_pairs,
        len(features_to_remove),
        len(final_descriptors)
    )

    # Save results
    if len(high_corr_pairs) > 0:
        save_high_corr_pairs(high_corr_pairs)
    save_final_descriptors(final_descriptors)

    print("\n✓ Correlation analysis complete!")

    return final_descriptors, high_corr_pairs
