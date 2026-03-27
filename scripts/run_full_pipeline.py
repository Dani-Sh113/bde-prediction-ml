"""
Run the complete BDE prediction ML pipeline from start to finish.

This script orchestrates all phases in sequence:
1. Correlation analysis
2. Model training
3. SHAP analysis
4. External validation
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config, data_loader, correlation_analysis, model_training, shap_analysis, validation, utils


def main():
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  BDE PREDICTION ML - COMPLETE PIPELINE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    pipeline_start = time.time()

    # Phase 1: Correlation Analysis
    print("\n")
    print("PHASE 1: CORRELATION ANALYSIS")
    print("-"*70)

    try:
        df = data_loader.load_full_dataset()
        data_loader.validate_data(df, verbose=True)
        final_descriptors, high_corr_pairs = correlation_analysis.run_correlation_analysis(df)
        print("✓ Phase 1 COMPLETE")
    except Exception as e:
        print(f"✗ Phase 1 FAILED: {e}")
        return

    # Phase 2: Model Training
    print("\n")
    print("PHASE 2: MODEL TRAINING")
    print("-"*70)

    try:
        X_train, X_test, y_train, y_test, feature_names = data_loader.load_and_prepare_training_data()
        model = model_training.run_model_training(X_train, X_test, y_train, y_test)
        print("✓ Phase 2 COMPLETE")
    except Exception as e:
        print(f"✗ Phase 2 FAILED: {e}")
        return

    # Phase 3: SHAP Analysis
    print("\n")
    print("PHASE 3: SHAP FEATURE IMPORTANCE")
    print("-"*70)

    try:
        feature_importance = shap_analysis.run_shap_analysis(
            model, X_train, X_test, y_test, feature_names
        )
        print("✓ Phase 3 COMPLETE")
    except Exception as e:
        print(f"✗ Phase 3 FAILED: {e}")
        return

    # Phase 4: External Validation
    print("\n")
    print("PHASE 4: EXTERNAL DATA VALIDATION")
    print("-"*70)

    try:
        X_external, y_external, ext_feature_names = data_loader.load_and_prepare_external_data()
        metrics = validation.run_external_validation(model, X_external, y_external, ext_feature_names)
        print("✓ Phase 4 COMPLETE")
    except Exception as e:
        print(f"✗ Phase 4 FAILED: {e}")
        return

    # Summary
    pipeline_end = time.time()
    total_time = pipeline_end - pipeline_start

    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PIPELINE COMPLETE - ALL PHASES SUCCESSFUL".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)

    print(f"\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    print(f"\nModel Performance (External Validation):")
    print(f"  R² Score:       {metrics['R² Score']:.4f}")
    print(f"  MAE:            {metrics['MAE (kJ/mol)']:.2f} kJ/mol")
    print(f"  RMSE:           {metrics['RMSE (kJ/mol)']:.2f} kJ/mol")
    print(f"  MAPE:           {metrics['MAPE (%)']:.2f}%")

    print(f"\nTop Features (SHAP):")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {int(row['Rank'])}. {row['Feature']:<20} {row['Contribution_%']:>6.1f}%")

    print(f"\nTotal Pipeline Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    print(f"\nGenerated Files:")
    print(f"  Model:                 {config.TRAINED_MODEL_PATH}")
    print(f"  Results Directory:     {config.RESULTS_DIR}")
    print(f"  Correlation Results:   {config.RESULTS_CORRELATION_DIR}")
    print(f"  Training Results:      {config.RESULTS_TRAINING_DIR}")
    print(f"  SHAP Results:          {config.RESULTS_SHAP_DIR}")
    print(f"  Validation Results:    {config.RESULTS_VALIDATION_DIR}")

    print("\n" + "="*70)
    print("Ready for deployment or publication!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
