"""
Enhanced Data Loader for LendingClub Pipeline
===========================================

This module provides enhanced data loading capabilities for the LendingClub
90-day delinquency prediction pipeline, including support for provided
pre-split datasets, data dictionary, and pre-trained models.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class LendingClubDataLoader:
    """Enhanced data loader for LendingClub pipeline with provided data files"""
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the data loader
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.train_data = None
        self.test_data = None
        self.data_dictionary = None
        self.model = None
        
        # File paths
        self.train_file = os.path.join(data_dir, 'train_lending_club.csv')
        self.test_file = os.path.join(data_dir, 'test_lending_club.csv')
        self.dict_file = os.path.join(data_dir, 'LCDataDictionary.csv')
        self.model_file = os.path.join(data_dir, 'model.joblib')
    
    def check_provided_files(self) -> Dict[str, bool]:
        """Check which provided files are available"""
        files_status = {
            'train_data': os.path.exists(self.train_file),
            'test_data': os.path.exists(self.test_file),
            'data_dictionary': os.path.exists(self.dict_file),
            'pretrained_model': os.path.exists(self.model_file)
        }
        
        print("Checking provided data files:")
        for file_type, exists in files_status.items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {file_type}")
        
        return files_status
    
    def load_data_dictionary(self) -> Optional[pd.DataFrame]:
        """Load the LendingClub data dictionary"""
        try:
            if not os.path.exists(self.dict_file):
                print(f"Data dictionary not found: {self.dict_file}")
                return None
            
            print("Loading LendingClub data dictionary...")
            self.data_dictionary = pd.read_csv(self.dict_file)
            print(f"Data dictionary loaded: {len(self.data_dictionary)} feature definitions")
            return self.data_dictionary
            
        except Exception as e:
            print(f"Error loading data dictionary: {e}")
            return None
    
    def load_train_test_data(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load pre-split train and test datasets"""
        try:
            print("Loading pre-split train and test datasets...")
            
            # Load training data
            if os.path.exists(self.train_file):
                self.train_data = pd.read_csv(self.train_file)
                print(f"Training data loaded: {self.train_data.shape}")
            else:
                print(f"Training file not found: {self.train_file}")
                return None, None
            
            # Load test data
            if os.path.exists(self.test_file):
                self.test_data = pd.read_csv(self.test_file)
                print(f"Test data loaded: {self.test_data.shape}")
            else:
                print(f"Test file not found: {self.test_file}")
                return None, None
            
            return self.train_data, self.test_data
            
        except Exception as e:
            print(f"Error loading train/test data: {e}")
            return None, None
    
    def load_pretrained_model(self):
        """Load the pre-trained model"""
        try:
            if not os.path.exists(self.model_file):
                print(f"Model file not found: {self.model_file}")
                return None
            
            print("Loading pre-trained model...")
            self.model = joblib.load(self.model_file)
            print(f"Model loaded successfully")
            print(f"Model type: {type(self.model).__name__}")
            
            # Try to get expected features
            if hasattr(self.model, 'feature_names_'):
                print(f"Expected features: {len(self.model.feature_names_)}")
            else:
                print("Expected features: Unknown")
            
            return self.model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'loan_status') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from the dataset with proper feature engineering
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Tuple of (features_df, target_series)
        """
        try:
            print("Preparing features and target...")
            
            # Make a copy to avoid modifying original
            df_processed = df.copy()
            
            # Feature engineering for date columns
            if 'issue_d' in df_processed.columns:
                df_processed['issue_d'] = pd.to_datetime(df_processed['issue_d'])
                df_processed['issue_d_year'] = df_processed['issue_d'].dt.year
                df_processed['issue_d_month'] = df_processed['issue_d'].dt.month
                df_processed = df_processed.drop('issue_d', axis=1)
            
            # Handle categorical encoding
            categorical_columns = df_processed.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != target_col:
                    # Simple label encoding for categorical variables
                    df_processed[col] = pd.Categorical(df_processed[col]).codes
            
            # Handle missing values
            numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col != target_col:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            
            # Separate features and target
            if target_col in df_processed.columns:
                target = df_processed[target_col]
                features = df_processed.drop(target_col, axis=1)
            else:
                target = None
                features = df_processed
            
            print(f"Features prepared: {features.shape}")
            print(f"Target prepared: {target.shape if target is not None else 'None'}")
            
            return features, target
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None, None
    
    def align_features_with_model(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align features with model expectations"""
        if self.model is None:
            return X
        
        try:
            # Get model's expected features
            if hasattr(self.model, 'feature_names_'):
                expected_features = self.model.feature_names_
                print(f"Aligning {len(X.columns)} features with {len(expected_features)} expected features")
                
                # Add missing features with zeros
                for feature in expected_features:
                    if feature not in X.columns:
                        X[feature] = 0
                
                # Reorder columns to match model expectations
                X = X[expected_features]
                print(f"Features aligned: {X.shape}")
            
            return X
            
        except Exception as e:
            print(f"Error aligning features: {e}")
            return X
    
    def get_feature_names(self) -> List[str]:
        """Get the expected feature names for the model"""
        if self.model and hasattr(self.model, 'feature_names_'):
            return list(self.model.feature_names_)
        else:
            # Default feature names based on the provided data structure
            return ['sub_grade', 'term', 'home_ownership', 'fico_range_low', 'total_acc', 
                   'pub_rec', 'revol_util', 'annual_inc', 'int_rate', 'dti', 'purpose',
                   'mort_acc', 'loan_amnt', 'application_type', 'installment', 
                   'verification_status', 'pub_rec_bankruptcies', 'addr_state',
                   'initial_list_status', 'fico_range_high', 'revol_bal', 'id',
                   'open_acc', 'emp_length', 'time_to_earliest_cr_line', 
                   'issue_d_year', 'issue_d_month']
    
    def create_feature_summary(self) -> Optional[pd.DataFrame]:
        """Create a comprehensive feature summary"""
        if self.train_data is None:
            print("No training data loaded")
            return None
        
        try:
            feature_info = []
            
            for col in self.train_data.columns:
                if col == 'loan_status':
                    continue
                
                info = {
                    'feature_name': col,
                    'dtype': str(self.train_data[col].dtype),
                    'missing_count': self.train_data[col].isnull().sum(),
                    'missing_pct': (self.train_data[col].isnull().sum() / len(self.train_data)) * 100,
                    'unique_values': self.train_data[col].nunique()
                }
                
                # Add description from data dictionary if available
                if self.data_dictionary is not None:
                    desc_row = self.data_dictionary[self.data_dictionary['LoanStatNew'] == col]
                    if not desc_row.empty:
                        info['description'] = desc_row.iloc[0]['Description']
                    else:
                        info['description'] = 'No description available'
                else:
                    info['description'] = 'Data dictionary not loaded'
                
                # Add statistics for numerical columns
                if pd.api.types.is_numeric_dtype(self.train_data[col]):
                    info.update({
                        'mean': self.train_data[col].mean(),
                        'std': self.train_data[col].std(),
                        'min': self.train_data[col].min(),
                        'max': self.train_data[col].max()
                    })
                else:
                    # For categorical columns, show top values
                    top_values = self.train_data[col].value_counts().head(3)
                    info['top_values'] = dict(top_values)
                
                feature_info.append(info)
            
            feature_summary = pd.DataFrame(feature_info)
            print(f"Feature summary created for {len(feature_summary)} features")
            return feature_summary
            
        except Exception as e:
            print(f"Error creating feature summary: {e}")
            return None
    
    def evaluate_pretrained_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the pre-trained model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            print("No model loaded")
            return {}
        
        try:
            print("Evaluating pre-trained model...")
            
            # Clean data - remove NaN values
            valid_mask = ~(pd.isna(y_test) | pd.isna(X_test).any(axis=1))
            X_test_clean = X_test[valid_mask]
            y_test_clean = y_test[valid_mask]
            
            print(f"Cleaned data: {len(X_test_clean)} samples (removed {len(X_test) - len(X_test_clean)} with NaN)")
            
            if len(X_test_clean) == 0:
                print("No valid samples after cleaning")
                return {}
            
            # Align features with model
            X_test_aligned = self.align_features_with_model(X_test_clean)
            
            # Make predictions
            if hasattr(self.model, 'predict_proba'):
                y_pred_proba = self.model.predict_proba(X_test_aligned)[:, 1]
            else:
                y_pred_proba = self.model.predict(X_test_aligned)
            
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import (
                roc_auc_score, precision_recall_curve, auc,
                accuracy_score, precision_score, recall_score, f1_score
            )
            
            metrics = {
                'accuracy': accuracy_score(y_test_clean, y_pred),
                'precision': precision_score(y_test_clean, y_pred),
                'recall': recall_score(y_test_clean, y_pred),
                'f1_score': f1_score(y_test_clean, y_pred),
                'roc_auc': roc_auc_score(y_test_clean, y_pred_proba)
            }
            
            # PR-AUC
            precision, recall, _ = precision_recall_curve(y_test_clean, y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
            
            print("Model Evaluation Results:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def get_model_feature_importance(self) -> Optional[pd.DataFrame]:
        """Extract feature importance from the loaded model"""
        if self.model is None:
            print("No model loaded")
            return None
        
        try:
            print("Extracting feature importance...")
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
            elif hasattr(self.model, 'get_feature_importance'):
                importance = self.model.get_feature_importance()
            else:
                print("Model doesn't have feature importance")
                return None
            
            # Get feature names
            feature_names = self.get_feature_names()
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(importance)],
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Add descriptions from data dictionary if available
            if self.data_dictionary is not None:
                importance_df['description'] = importance_df['feature'].apply(
                    lambda x: self._get_feature_description(x)
                )
            
            print(f"Feature importance extracted for {len(importance_df)} features")
            return importance_df
            
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None
    
    def _get_feature_description(self, feature_name: str) -> str:
        """Get feature description from data dictionary"""
        if self.data_dictionary is None:
            return "No description available"
        
        desc_row = self.data_dictionary[self.data_dictionary['LoanStatNew'] == feature_name]
        if not desc_row.empty:
            return desc_row.iloc[0]['Description']
        else:
            return "Description not found"

def main():
    """Demo function for the enhanced data loader"""
    print("=" * 60)
    print("ENHANCED DATA LOADER DEMONSTRATION")
    print("=" * 60)
    
    # Initialize loader
    loader = LendingClubDataLoader()
    
    # Check provided files
    files_status = loader.check_provided_files()
    
    if not all(files_status.values()):
        print("\nSome required files are missing. Please ensure all data files are in the data/ directory.")
        return
    
    # Load data dictionary
    data_dict = loader.load_data_dictionary()
    
    # Load train/test data
    train_df, test_df = loader.load_train_test_data()
    
    if train_df is not None and test_df is not None:
        print(f"\nDataset Overview:")
        print(f"  Training samples: {len(train_df):,}")
        print(f"  Test samples: {len(test_df):,}")
        print(f"  Total features: {len(train_df.columns) - 1}")  # -1 for target
        
        # Prepare features
        X_train, y_train = loader.prepare_features(train_df)
        X_test, y_test = loader.prepare_features(test_df)
        
        # Load and evaluate model
        model = loader.load_pretrained_model()
        if model is not None:
            metrics = loader.evaluate_pretrained_model(X_test, y_test)
            
            # Get feature importance
            importance_df = loader.get_model_feature_importance()
            if importance_df is not None:
                print(f"\nTop 10 Most Important Features:")
                print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
        
        # Create feature summary
        feature_summary = loader.create_feature_summary()
        if feature_summary is not None:
            print(f"\nFeature Summary (first 5):")
            print(feature_summary[['feature_name', 'dtype', 'missing_pct', 'unique_values']].head().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("DATA LOADER DEMONSTRATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
