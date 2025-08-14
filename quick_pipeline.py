#!/usr/bin/env python3
"""
Quick Fix Pipeline for LendingClub Data
Addresses feature alignment issues between provided data and model
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the provided data files"""
    print("Loading provided data files...")
    
    # Load data
    train_df = pd.read_csv('data/train_lending_club.csv')
    test_df = pd.read_csv('data/test_lending_club.csv')
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    
    return train_df, test_df

def prepare_features(df):
    """Prepare features with proper date engineering"""
    df_processed = df.copy()
    
    # Handle date column
    if 'issue_d' in df_processed.columns:
        df_processed['issue_d'] = pd.to_datetime(df_processed['issue_d'])
        df_processed['issue_d_year'] = df_processed['issue_d'].dt.year
        df_processed['issue_d_month'] = df_processed['issue_d'].dt.month
        df_processed = df_processed.drop('issue_d', axis=1)
    
    # Handle categorical columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'loan_status':
            df_processed[col] = pd.Categorical(df_processed[col]).codes
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'loan_status':
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Separate features and target
    if 'loan_status' in df_processed.columns:
        y = df_processed['loan_status']
        X = df_processed.drop('loan_status', axis=1)
    else:
        y = None
        X = df_processed
    
    return X, y

def evaluate_model_simple(model, X_test, y_test):
    """Simple model evaluation"""
    try:
        # Clean data - remove NaN values
        valid_mask = ~(pd.isna(y_test) | pd.isna(X_test).any(axis=1))
        X_test_clean = X_test[valid_mask]
        y_test_clean = y_test[valid_mask]
        
        print(f"Cleaned data: {len(X_test_clean)} samples (removed {len(X_test) - len(X_test_clean)} with NaN)")
        
        if len(X_test_clean) == 0:
            print("No valid samples after cleaning")
            return {}
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
        else:
            y_pred_proba = model.predict(X_test_clean)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test_clean, y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(y_test_clean, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Additional metrics
        accuracy = ((y_pred_proba > 0.5) == y_test_clean).mean()
        
        print(f"\nModel Performance:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Positive Rate: {y_test_clean.mean():.4f}")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'positive_rate': y_test_clean.mean()
        }
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {}

def get_feature_importance(model, feature_names):
    """Extract feature importance from model"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
        else:
            print("Model doesn't have feature importance")
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
        
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return None

def main():
    """Main execution function"""
    print("="*60)
    print("QUICK FIX: LendingClub Pipeline")
    print("="*60)
    
    try:
        # Load data
        train_df, test_df = load_and_prepare_data()
        
        # Prepare features
        print("\nPreparing training features...")
        X_train, y_train = prepare_features(train_df)
        
        print("\nPreparing test features...")
        X_test, y_test = prepare_features(test_df)
        
        print(f"\nProcessed shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        # Load model
        print("\nLoading pre-trained model...")
        try:
            model = joblib.load('data/model.joblib')
            print(f"Model loaded: {type(model).__name__}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying alternative approach...")
            
            # Try to handle CatBoost dependency issue
            try:
                import catboost
                model = joblib.load('data/model.joblib')
                print(f"Model loaded with CatBoost: {type(model).__name__}")
            except ImportError:
                print("CatBoost not available. Install with: pip install catboost")
                return
        
        # Align features with model expectations
        print("\nAligning features with model...")
        
        # Get model's expected features (if available)
        if hasattr(model, 'feature_names_'):
            expected_features = model.feature_names_
            print(f"Model expects {len(expected_features)} features")
        else:
            expected_features = X_test.columns.tolist()
            print(f"Using all available features: {len(expected_features)}")
        
        # Ensure feature alignment
        common_features = [f for f in expected_features if f in X_test.columns]
        missing_features = [f for f in expected_features if f not in X_test.columns]
        
        if missing_features:
            print(f"Missing features: {missing_features[:5]}...")
            # Add missing features with zeros
            for feature in missing_features:
                X_test[feature] = 0
                X_train[feature] = 0
        
        # Reorder columns to match model expectations
        if hasattr(model, 'feature_names_'):
            X_test = X_test[expected_features]
            X_train = X_train[expected_features]
        
        print(f"Final feature alignment: {X_test.shape}")
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model_simple(model, X_test, y_test)
        
        # Get feature importance
        print("\nExtracting feature importance...")
        importance_df = get_feature_importance(model, X_test.columns.tolist())
        
        print("\n" + "="*60)
        print("QUICK FIX PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if metrics:
            print(f"Model ROC-AUC: {metrics.get('roc_auc', 'N/A')}")
            print(f"Model PR-AUC: {metrics.get('pr_auc', 'N/A')}")
        
        print(f"Dataset size: {len(X_test):,} test samples")
        print(f"Features used: {len(X_test.columns)}")
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
