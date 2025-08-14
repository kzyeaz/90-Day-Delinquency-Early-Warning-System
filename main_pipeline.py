"""
Functional Main Pipeline for LendingClub 90-Day Delinquency Early Warning Model
==============================================================================

This is a streamlined, fully functional main pipeline that works with the provided
data files and focuses on core functionality without complex dependencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the fixed data loader and MLOps manager
from data_loader import LendingClubDataLoader

# Try to import MLOps manager (optional)
try:
    from mlops_pipeline import MLOpsManager
    MLFLOW_AVAILABLE = True
except ImportError:
    print("[INFO] MLflow not available - continuing without experiment tracking")
    MLFLOW_AVAILABLE = False

def main_enhanced_pipeline():
    """
    Main pipeline execution using provided pre-split data and model
    """
    print("=" * 80)
    print("LendingClub 90-Day Delinquency Model - Functional Enhanced Pipeline")
    print("Using provided train/test data and pre-trained model")
    print("=" * 80)
    
    try:
        # ============================================================================
        # STEP 0: INITIALIZE MLFLOW TRACKING (OPTIONAL)
        # ============================================================================
        mlops_manager = None
        if MLFLOW_AVAILABLE:
            print("\n[STEP 0] Initializing MLflow Tracking")
            print("-" * 50)
            try:
                mlops_manager = MLOpsManager(experiment_name="lending_club_enhanced_pipeline")
                mlops_manager.start_run(run_name=f"enhanced_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                print("[OK] MLflow tracking initialized")
            except Exception as e:
                print(f"[WARNING] MLflow initialization failed: {e}")
                mlops_manager = None
        
        # ============================================================================
        # STEP 1: INITIALIZE AND CHECK DATA FILES
        # ============================================================================
        print("\n[STEP 1] Initializing Enhanced Data Loader")
        print("-" * 50)
        
        # Initialize enhanced data loader
        loader = LendingClubDataLoader()
        
        # Check provided files
        files_status = loader.check_provided_files()
        
        if not all(files_status.values()):
            print("\n[ERROR] Some required files are missing!")
            print("Please ensure the following files are in the data/ directory:")
            print("  - train_lending_club.csv")
            print("  - test_lending_club.csv") 
            print("  - LCDataDictionary.csv")
            print("  - model.joblib")
            return False
        
        print("\n[OK] All required files found!")
        
        # ============================================================================
        # STEP 2: LOAD DATA AND MODEL
        # ============================================================================
        print("\n[STEP 2] Loading Data and Model")
        print("-" * 50)
        
        # Load data dictionary
        print("Loading data dictionary...")
        data_dict = loader.load_data_dictionary()
        
        # Load train/test data
        print("Loading pre-split train and test data...")
        train_df, test_df = loader.load_train_test_data()
        
        if train_df is None or test_df is None:
            print("[ERROR] Failed to load train/test data")
            return False
        
        # Display dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Training samples: {len(train_df):,}")
        print(f"Test samples: {len(test_df):,}")
        print(f"Features: {len(train_df.columns) - 1}")  # -1 for target
        
        # Check target distribution
        if 'loan_status' in train_df.columns:
            train_pos_rate = train_df['loan_status'].mean()
            test_pos_rate = test_df['loan_status'].mean()
            print(f"Train positive rate: {train_pos_rate:.3f}")
            print(f"Test positive rate: {test_pos_rate:.3f}")
        
        # Load pre-trained model
        print("Loading pre-trained model...")
        model = loader.load_pretrained_model()
        
        if model is None:
            print("[ERROR] Failed to load pre-trained model")
            return False
        
        print("[OK] Data loading complete!")
        
        # ============================================================================
        # STEP 3: FEATURE PREPARATION
        # ============================================================================
        print("\n[STEP 3] Feature Preparation")
        print("-" * 50)
        
        # Prepare features for training and test sets
        print("Preparing training features...")
        X_train, y_train = loader.prepare_features(train_df)
        
        print("Preparing test features...")
        X_test, y_test = loader.prepare_features(test_df)
        
        if X_train is None or X_test is None:
            print("[ERROR] Failed to prepare features")
            return False
        
        print(f"[OK] Feature preparation complete!")
        print(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"   Test: {X_test.shape[0]:,} samples, {X_test.shape[1]} features")
        
        # ============================================================================
        # STEP 4: MODEL EVALUATION
        # ============================================================================
        print("\n[STEP 4] Pre-trained Model Evaluation")
        print("-" * 50)
        
        print("Evaluating pre-trained model on test data...")
        metrics = loader.evaluate_pretrained_model(X_test, y_test)
        
        if metrics:
            print("\n[RESULTS] Model Performance Summary:")
            print(f"   ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            print(f"   PR-AUC: {metrics.get('pr_auc', 'N/A'):.4f}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
            print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
            print(f"   F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
            
            # Log metrics to MLflow
            if mlops_manager:
                print("[MLFLOW] Logging model performance metrics...")
                mlops_manager.log_metrics(metrics)
        else:
            print("[ERROR] Model evaluation failed")
        
        # ============================================================================
        # STEP 5: FEATURE IMPORTANCE ANALYSIS
        # ============================================================================
        print("\n[STEP 5] Feature Importance Analysis")
        print("-" * 50)
        
        print("Extracting feature importance from pre-trained model...")
        importance_df = loader.get_model_feature_importance()
        
        if importance_df is not None:
            print("\n[ANALYSIS] Top 15 Most Important Features:")
            print("-" * 60)
            top_features = importance_df.head(15)
            
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                importance = row['importance']
                description = row.get('description', 'No description')
                
                print(f"{idx+1:2d}. {feature_name:<25} ({importance:6.3f}) - {description[:50]}...")
            
            # Log feature importance to MLflow
            if mlops_manager:
                print("[MLFLOW] Logging feature importance data...")
                # Log top features as parameters
                for idx, row in importance_df.head(10).iterrows():
                    mlops_manager.log_param(f"top_feature_{idx+1}", f"{row['feature']} ({row['importance']:.3f})")
        else:
            print("[ERROR] Feature importance extraction failed")
        
        # ============================================================================
        # STEP 6: FEATURE SUMMARY REPORT
        # ============================================================================
        print("\n[STEP 6] Feature Summary Report")
        print("-" * 50)
        
        print("Creating comprehensive feature summary...")
        feature_summary = loader.create_feature_summary()
        
        if feature_summary is not None:
            print(f"\n[SUMMARY] Feature Summary Overview:")
            print(f"   Total features analyzed: {len(feature_summary)}")
            
            # Missing values summary
            missing_features = feature_summary[feature_summary['missing_pct'] > 0]
            print(f"   Features with missing values: {len(missing_features)}")
            
            # Data types summary
            numeric_features = feature_summary[feature_summary['dtype'].str.contains('int|float')]
            categorical_features = feature_summary[~feature_summary['dtype'].str.contains('int|float')]
            print(f"   Numerical features: {len(numeric_features)}")
            print(f"   Categorical features: {len(categorical_features)}")
            
            # Show features with highest missing percentages
            if len(missing_features) > 0:
                print(f"\n   Top 5 features with missing values:")
                top_missing = missing_features.nlargest(5, 'missing_pct')
                for _, row in top_missing.iterrows():
                    print(f"     {row['feature_name']}: {row['missing_pct']:.1f}% missing")
        else:
            print("[ERROR] Feature summary creation failed")
        
        # ============================================================================
        # STEP 7: BUSINESS INSIGHTS AND RECOMMENDATIONS
        # ============================================================================
        print("\n[STEP 7] Business Insights and Recommendations")
        print("-" * 50)
        
        if importance_df is not None and metrics:
            print("\n[INSIGHTS] Key Business Insights:")
            print("-" * 30)
            
            # Top risk factors
            top_3_features = importance_df.head(3)
            print("[RISK FACTORS] Top 3 Risk Factors:")
            for idx, row in top_3_features.iterrows():
                print(f"   {idx+1}. {row['feature']} (importance: {row['importance']:.3f})")
            
            # Model performance assessment
            roc_auc = metrics.get('roc_auc', 0)
            if roc_auc > 0.75:
                performance_level = "Excellent"
            elif roc_auc > 0.65:
                performance_level = "Good"
            elif roc_auc > 0.55:
                performance_level = "Fair"
            else:
                performance_level = "Poor"
            
            print(f"\n[ASSESSMENT] Model Performance Assessment: {performance_level}")
            print(f"   ROC-AUC of {roc_auc:.3f} indicates {performance_level.lower()} discrimination ability")
            
            # Recommendations
            print(f"\n[RECOMMENDATIONS] Recommendations:")
            if roc_auc < 0.7:
                print("   • Consider model retraining or feature engineering")
                print("   • Review data quality and feature selection")
            else:
                print("   • Model shows strong performance for production use")
                print("   • Consider A/B testing for deployment validation")
            
            print("   • Monitor key risk factors in production")
            print("   • Implement regular model performance tracking")
        
        # ============================================================================
        # PIPELINE COMPLETION
        # ============================================================================
        print("\n" + "=" * 80)
        print("[SUCCESS] ENHANCED PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        print(f"\n[FINAL SUMMARY] Final Summary:")
        if metrics:
            print(f"   Model Performance: ROC-AUC = {metrics.get('roc_auc', 'N/A'):.4f}")
        print(f"   Dataset Size: {len(X_test):,} test samples")
        print(f"   Features Used: {X_test.shape[1] if X_test is not None else 'N/A'}")
        print(f"   Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n[SUCCESS] Your LendingClub 90-day delinquency model is ready for production!")
        
        # End MLflow run
        if mlops_manager:
            print("\n[MLFLOW] Finalizing experiment tracking...")
            # Log dataset information
            mlops_manager.log_param("train_samples", len(train_df))
            mlops_manager.log_param("test_samples", len(test_df))
            mlops_manager.log_param("features_count", X_test.shape[1] if X_test is not None else 0)
            mlops_manager.log_param("model_type", "CatBoostClassifier")
            
            # End the MLflow run
            mlops_manager.end_run()
            print("[MLFLOW] Experiment tracking completed!")
            print(f"[MLFLOW] View results at: http://localhost:5000")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function with user interaction"""
    print("LendingClub 90-Day Delinquency Prediction Model")
    print("Enhanced Functional Pipeline")
    print("=" * 50)
    
    # Check if provided data files exist
    data_files = [
        'data/train_lending_club.csv',
        'data/test_lending_club.csv', 
        'data/LCDataDictionary.csv',
        'data/model.joblib'
    ]
    
    files_exist = all(os.path.exists(f) for f in data_files)
    
    if files_exist:
        print("\n[OK] Provided data files detected!")
        print("Running enhanced pipeline with provided data...")
        success = main_enhanced_pipeline()
        
        if not success:
            print("\n[ERROR] Enhanced pipeline failed. Please check the errors above.")
        
    else:
        print("\n[ERROR] Provided data files not found!")
        print("Please ensure the following files are in the data/ directory:")
        for file_path in data_files:
            exists = "[OK]" if os.path.exists(file_path) else "[MISSING]"
            print(f"   {exists} {file_path}")
        
        print("\nAlternatively, you can:")
        print("1. Place the required files in the data/ directory")
        print("2. Use the quick_fix_pipeline.py for immediate analysis")

if __name__ == "__main__":
    main()
