"""
Run correlation analysis on the full dataset.

This script:
1. Loads the full dataset
2. Performs correlation analysis
3. Identifies and visualizes highly correlated features
4. Generates the final descriptor list
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, data_loader, correlation_analysis


def main():
    print("="*70)
    print("BDE PREDICTION ML - CORRELATION ANALYSIS")
    print("="*70)

    # Load data
    df = data_loader.load_full_dataset()

    # Validate data
    data_loader.validate_data(df, verbose=True)

    # Run correlation analysis
    final_descriptors, high_corr_pairs = correlation_analysis.run_correlation_analysis(df)

    print("\n" + "="*70)
    print("COMPLETE - READY FOR MODEL TRAINING")
    print("="*70)
    print(f"\nNext step: Run 'python scripts/run_model_training.py'")


if __name__ == "__main__":
    main()
