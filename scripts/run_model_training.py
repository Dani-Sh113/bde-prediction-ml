"""
Train TPOT model on the dataset.

This script:
1. Loads and preprocesses the training data
2. Trains a TPOT regressor
3. Evaluates performance on test set
4. Saves trained model and visualizations
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, data_loader, model_training


def main():
    print("="*70)
    print("BDE PREDICTION ML - MODEL TRAINING")
    print("="*70)

    # Load and prepare data
    print("\nLoading training data...")
    X_train, X_test, y_train, y_test, feature_names = data_loader.load_and_prepare_training_data()

    # Train model
    print("\nTraining TPOT model...")
    model = model_training.run_model_training(X_train, X_test, y_train, y_test)

    print("\n" + "="*70)
    print("COMPLETE - READY FOR FEATURE ANALYSIS")
    print("="*70)
    print(f"\nModel saved to: {config.TRAINED_MODEL_PATH}")
    print(f"Next step: Run 'python scripts/run_shap_analysis.py'")


if __name__ == "__main__":
    main()
