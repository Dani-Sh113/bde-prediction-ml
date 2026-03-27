"""
Perform SHAP feature importance analysis.

This script:
1. Loads the trained model
2. Loads test data
3. Calculates SHAP values
4. Generates feature importance analysis and visualizations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, data_loader, utils, shap_analysis


def main():
    print("="*70)
    print("BDE PREDICTION ML - SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test, feature_names = data_loader.load_and_prepare_training_data()

    # Load trained model
    print("Loading trained model...")
    model = utils.load_model()

    # Run SHAP analysis
    print("\nPerforming SHAP analysis...")
    feature_importance = shap_analysis.run_shap_analysis(
        model, X_train, X_test, y_test, feature_names
    )

    print("\n" + "="*70)
    print("COMPLETE - READY FOR VALIDATION")
    print("="*70)
    print(f"\nTop 3 features:")
    for idx, row in feature_importance.head(3).iterrows():
        print(f"  {int(row['Rank'])}. {row['Feature']} ({row['Contribution_%']:.1f}%)")
    print(f"\nNext step: Run 'python scripts/run_validation.py'")


if __name__ == "__main__":
    main()
