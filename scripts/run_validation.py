"""
Validate model on external unseen dataset.

This script:
1. Loads the trained model
2. Loads external validation data
3. Generates predictions
4. Evaluates performance and generates visualizations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, data_loader, utils, validation


def main():
    print("="*70)
    print("BDE PREDICTION ML - EXTERNAL DATA VALIDATION")
    print("="*70)

    # Load trained model
    print("Loading trained model...")
    model = utils.load_model()

    # Load external data
    print("Loading external validation data...")
    X_external, y_external, feature_names = data_loader.load_and_prepare_external_data()

    # Run validation
    print("\nPerforming external validation...")
    metrics = validation.run_external_validation(model, X_external, y_external, feature_names)

    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    print(f"\nFinal Model Performance:")
    print(f"  R² Score: {metrics['R² Score']:.4f}")
    print(f"  MAE: {metrics['MAE (kJ/mol)']:.2f} kJ/mol")
    print(f"  RMSE: {metrics['RMSE (kJ/mol)']:.2f} kJ/mol")
    print(f"\nResults saved to: {config.RESULTS_VALIDATION_DIR}")


if __name__ == "__main__":
    main()
