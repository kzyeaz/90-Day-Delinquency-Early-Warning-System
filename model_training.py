"""
Model Training Module for LendingClub 90-Day Delinquency Prediction
===================================================================

This module handles model training, hyperparameter tuning, and calibration
with monotonic constraints and proper evaluation metrics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import lightgbm as lgb
import xgboost as xgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
from typing import Dict, Tuple, List, Any
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    Handles model training with monotonic constraints and proper evaluation
    """
    
    def __init__(self):
        self.feature_columns = None
        self.categorical_features = None
        self.monotonic_constraints = None
        self.label_encoders = {}
        self.scaler = None
        
    def prepare_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                        test_df: pd.DataFrame, target_col: str) -> Tuple[Dict, Dict]:
        """
        Prepare features for training with proper encoding and constraints
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame  
            test_df: Test DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (data_dict, metadata_dict)
        """
        print("Preparing features for training...")
        
        # Define features to exclude (non-predictive or leakage risk)
        exclude_cols = [
            target_col, 'loan_status', 'issue_d', 'earliest_cr_line',
            'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d',
            'id', 'member_id', 'url', 'desc', 'title', 'zip_code',
            'emp_title', 'addr_state'  # Remove for now, can add back with encoding
        ]
        
        # Get feature columns
        all_cols = train_df.columns.tolist()
        feature_cols = [col for col in all_cols if col not in exclude_cols]
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in feature_cols:
            if train_df[col].dtype == 'object' or train_df[col].nunique() < 20:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        
        # Prepare datasets
        datasets = {'train': train_df, 'val': val_df, 'test': test_df}
        prepared_data = {}
        
        for split_name, df in datasets.items():
            df_prep = df.copy()
            
            # Handle missing values
            for col in numerical_features:
                if col in df_prep.columns:
                    df_prep[col] = df_prep[col].fillna(df_prep[col].median())
            
            for col in categorical_features:
                if col in df_prep.columns:
                    df_prep[col] = df_prep[col].fillna('Unknown')
            
            # Encode categorical features
            for col in categorical_features:
                if col in df_prep.columns:
                    if split_name == 'train':
                        # Fit encoder on training data
                        le = LabelEncoder()
                        df_prep[col] = le.fit_transform(df_prep[col].astype(str))
                        self.label_encoders[col] = le
                    else:
                        # Transform using fitted encoder
                        le = self.label_encoders[col]
                        # Handle unseen categories
                        df_prep[col] = df_prep[col].astype(str)
                        mask = df_prep[col].isin(le.classes_)
                        df_prep.loc[~mask, col] = 'Unknown'
                        
                        # Add 'Unknown' to encoder if not present
                        if 'Unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'Unknown')
                        
                        df_prep[col] = le.transform(df_prep[col])
            
            # Select final feature columns
            final_features = [col for col in feature_cols if col in df_prep.columns]
            
            prepared_data[split_name] = {
                'X': df_prep[final_features],
                'y': df_prep[target_col] if target_col in df_prep.columns else None
            }
        
        # Define monotonic constraints for key features
        monotonic_constraints = {}
        constraint_features = {
            'dti': 1,  # Higher DTI = higher risk
            'delinq_2yrs': 1,  # More delinquencies = higher risk
            'inq_last_6mths': 1,  # More inquiries = higher risk
            'revol_util': 1,  # Higher utilization = higher risk
            'annual_inc': -1,  # Higher income = lower risk
            'payment_to_income': 1,  # Higher payment burden = higher risk
            'loan_to_income': 1,  # Higher loan ratio = higher risk
            'credit_age_months': -1,  # Longer credit history = lower risk
            'emp_length_numeric': -1,  # Longer employment = lower risk
            'int_rate': 1,  # Higher rate = higher risk (already priced in)
        }
        
        # Map constraints to feature indices
        for feature, constraint in constraint_features.items():
            if feature in final_features:
                idx = final_features.index(feature)
                monotonic_constraints[idx] = constraint
        
        self.feature_columns = final_features
        self.categorical_features = [col for col in categorical_features if col in final_features]
        self.monotonic_constraints = monotonic_constraints
        
        metadata = {
            'feature_columns': final_features,
            'categorical_features': self.categorical_features,
            'monotonic_constraints': monotonic_constraints,
            'label_encoders': self.label_encoders
        }
        
        print(f"Final feature count: {len(final_features)}")
        print(f"Monotonic constraints: {len(monotonic_constraints)}")
        
        return prepared_data, metadata
    
    def train_lightgbm(self, train_data: Dict, val_data: Dict, 
                      params: Dict = None) -> lgb.LGBMClassifier:
        """
        Train LightGBM model with monotonic constraints
        
        Args:
            train_data: Training data dict with X, y
            val_data: Validation data dict with X, y
            params: Optional hyperparameters
            
        Returns:
            Trained LightGBM model
        """
        print("Training LightGBM model...")
        
        # Default parameters optimized for credit risk
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 100,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        # Handle class imbalance
        pos_weight = (train_data['y'] == 0).sum() / (train_data['y'] == 1).sum()
        default_params['scale_pos_weight'] = pos_weight
        
        # Create model with monotonic constraints
        model = lgb.LGBMClassifier(**default_params)
        
        # Prepare monotonic constraints for LightGBM
        if self.monotonic_constraints:
            # LightGBM expects constraints as a list
            constraint_list = [0] * len(self.feature_columns)
            for idx, constraint in self.monotonic_constraints.items():
                if idx < len(constraint_list):
                    constraint_list[idx] = constraint
            model.set_params(monotone_constraints=constraint_list)
        
        # Train model
        model.fit(
            train_data['X'], train_data['y'],
            eval_set=[(val_data['X'], val_data['y'])],
            eval_names=['validation'],
            categorical_feature=self.categorical_features
        )
        
        return model
    
    def train_xgboost(self, train_data: Dict, val_data: Dict,
                     params: Dict = None) -> xgb.XGBClassifier:
        """
        Train XGBoost model with monotonic constraints
        
        Args:
            train_data: Training data dict with X, y
            val_data: Validation data dict with X, y  
            params: Optional hyperparameters
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost model...")
        
        # Default parameters
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_estimators': 1000,
            'early_stopping_rounds': 100
        }
        
        if params:
            default_params.update(params)
        
        # Handle class imbalance
        pos_weight = (train_data['y'] == 0).sum() / (train_data['y'] == 1).sum()
        default_params['scale_pos_weight'] = pos_weight
        
        # Create model
        model = xgb.XGBClassifier(**default_params)
        
        # Set monotonic constraints for XGBoost
        if self.monotonic_constraints:
            # XGBoost expects constraints as a tuple
            constraint_list = [0] * len(self.feature_columns)
            for idx, constraint in self.monotonic_constraints.items():
                if idx < len(constraint_list):
                    constraint_list[idx] = constraint
            model.set_params(monotone_constraints=tuple(constraint_list))
        
        # Train model
        model.fit(
            train_data['X'], train_data['y'],
            eval_set=[(val_data['X'], val_data['y'])],
            verbose=False
        )
        
        return model
    
    def calibrate_model(self, model, train_data: Dict, val_data: Dict, 
                       method: str = 'isotonic') -> CalibratedClassifierCV:
        """
        Calibrate model probabilities
        
        Args:
            model: Trained model
            train_data: Training data
            val_data: Validation data
            method: Calibration method ('isotonic' or 'sigmoid')
            
        Returns:
            Calibrated model
        """
        print(f"Calibrating model using {method} method...")
        
        # Use validation set for calibration
        calibrated_model = CalibratedClassifierCV(
            model, method=method, cv='prefit'
        )
        
        calibrated_model.fit(val_data['X'], val_data['y'])
        
        return calibrated_model
    
    def evaluate_model(self, model, test_data: Dict, 
                      model_name: str = "Model") -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            test_data: Test data dict
            model_name: Name for reporting
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred_proba = model.predict_proba(test_data['X'])[:, 1]
        y_pred = model.predict(test_data['X'])
        y_true = test_data['y']
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # KS Statistic
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        ks_stat = max(tpr - fpr)
        
        # Brier Score (calibration)
        brier_score = brier_score_loss(y_true, y_pred_proba)
        
        # Classification metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'ks_statistic': ks_stat,
            'brier_score': brier_score,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
        
        print(f"\n{model_name} Performance:")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"KS Statistic: {metrics['ks_statistic']:.4f}")
        print(f"Brier Score: {metrics['brier_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def plot_calibration_curve(self, model, test_data: Dict, 
                              model_name: str = "Model"):
        """
        Plot calibration curve for the model
        
        Args:
            model: Trained model
            test_data: Test data
            model_name: Model name for plot title
        """
        y_pred_proba = model.predict_proba(test_data['X'])[:, 1]
        y_true = test_data['y']
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label=f"{model_name}")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def hyperparameter_tuning(trainer: ModelTrainer, train_data: Dict, 
                         val_data: Dict, model_type: str = 'lightgbm') -> Dict:
    """
    Perform hyperparameter tuning
    
    Args:
        trainer: ModelTrainer instance
        train_data: Training data
        val_data: Validation data
        model_type: 'lightgbm' or 'xgboost'
        
    Returns:
        Best parameters
    """
    print(f"Performing hyperparameter tuning for {model_type}...")
    
    if model_type == 'lightgbm':
        param_grid = {
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 30]
        }
        
        base_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            random_state=42,
            n_estimators=100,  # Reduced for tuning speed
            verbose=-1
        )
        
    else:  # xgboost
        param_grid = {
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_estimators=100  # Reduced for tuning speed
        )
    
    # Grid search with cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, 
        cv=cv, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(train_data['X'], train_data['y'])
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_
