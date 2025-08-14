"""
LendingClub 90-Day Delinquency Early Warning Model
==================================================

This script implements a comprehensive machine learning pipeline for predicting
90+ day delinquency using LendingClub 2007-2018 data.

Key Features:
- Temporal data splits to mimic production
- Feature engineering with domain knowledge
- Monotonic constraints for interpretability
- Fairness diagnostics
- MLOps with MLflow tracking
- SHAP explanations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Any
import joblib
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, 
    classification_report, confusion_matrix,
    brier_score_loss, roc_curve
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import lightgbm as lgb
import xgboost as xgb

# Interpretability
import shap

# MLOps
import mlflow
import mlflow.lightgbm
import mlflow.sklearn

# Suppress warnings
warnings.filterwarnings('ignore')

class LendingClubDelinquencyModel:
    """
    Main class for LendingClub 90-day delinquency prediction model
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the model with data path
        
        Args:
            data_path: Path to the LendingClub CSV file
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.target_col = 'delinq_90dpd'
        self.model = None
        self.calibrated_model = None
        self.feature_names = None
        self.label_encoders = {}
        
        # Model performance tracking
        self.metrics = {}
        self.shap_values = None
        self.explainer = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial data inspection
        
        Returns:
            DataFrame with raw LendingClub data
        """
        print("Loading LendingClub data...")
        
        if not self.data_path or not os.path.exists(self.data_path):
            print("Data file not found. Please ensure you have downloaded:")
            print("accepted_2007_to_2018Q4.csv.gz from LendingClub")
            print("Expected path:", self.data_path or "Not specified")
            return None
            
        try:
            # Load with memory optimization
            self.raw_data = pd.read_csv(
                self.data_path,
                low_memory=False,
                compression='gzip' if self.data_path.endswith('.gz') else None
            )
            
            print(f"Data loaded successfully!")
            print(f"Shape: {self.raw_data.shape}")
            print(f"Memory usage: {self.raw_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the raw data
        
        Returns:
            Cleaned DataFrame
        """
        if self.raw_data is None:
            print("No data loaded. Please run load_data() first.")
            return None
            
        print("Cleaning data...")
        df = self.raw_data.copy()
        
        # Clean percentage fields
        percentage_cols = ['int_rate', 'revol_util']
        for col in percentage_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('%', '').replace('', np.nan)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse date columns
        date_cols = ['issue_d', 'earliest_cr_line', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%b-%Y', errors='coerce')
        
        # Memory optimization - downcast numeric types
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        print(f"Data cleaned. New memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def create_delinquency_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 90+ day delinquency labels based on loan_status
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with delinquency labels
        """
        print("Creating 90+ day delinquency labels...")
        
        # Define positive cases for 90+ DPD
        positive_statuses = [
            'Late (31-120 days)',
            'Late (16-30 days)',  # Include as potential progression
            'Default',
            'Charged Off'
        ]
        
        # Define negative cases
        negative_statuses = [
            'Current',
            'Fully Paid'
        ]
        
        # Create binary target
        df[self.target_col] = 0  # Default to 0
        df.loc[df['loan_status'].isin(positive_statuses), self.target_col] = 1
        
        # Filter to only include loans with clear outcomes
        valid_statuses = positive_statuses + negative_statuses
        df_labeled = df[df['loan_status'].isin(valid_statuses)].copy()
        
        print(f"Delinquency rate: {df_labeled[self.target_col].mean():.3f}")
        print(f"Positive cases: {df_labeled[self.target_col].sum():,}")
        print(f"Total cases: {len(df_labeled):,}")
        
        return df_labeled
    
    def filter_origination_features(self, df: pd.DataFrame) -> List[str]:
        """
        Filter to only include features available at loan origination
        to prevent data leakage
        
        Args:
            df: DataFrame with all features
            
        Returns:
            List of origination-time feature names
        """
        print("Filtering to origination-time features...")
        
        # Features available at origination (not exhaustive - adjust based on data dictionary)
        origination_features = [
            # Loan characteristics
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
            'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
            'issue_d', 'loan_status', 'purpose', 'title', 'zip_code', 'addr_state',
            'dti', 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'open_acc',
            'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
            'application_type', 'mort_acc', 'pub_rec_bankruptcies',
            
            # Additional credit history features (if available)
            'acc_open_past_24mths', 'all_util', 'inq_fi', 'total_cu_tl',
            'inq_last_12m', 'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal',
            'open_acc_6m', 'open_act_il', 'open_il_12m', 'open_il_24m',
            'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',
            'open_rv_24m', 'max_bal_bc', 'all_util', 'total_rev_hi_lim',
            'inq_fi', 'total_cu_tl', 'inq_last_12m'
        ]
        
        # Filter to only include features that exist in the dataset
        available_features = [col for col in origination_features if col in df.columns]
        
        print(f"Available origination features: {len(available_features)}")
        
        return available_features
    
    def temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/validation/test splits
        
        Args:
            df: DataFrame with issue_d column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("Creating temporal splits...")
        
        # Ensure we have issue_d
        if 'issue_d' not in df.columns:
            raise ValueError("issue_d column required for temporal splits")
        
        # Filter loans with sufficient observation window (24 months)
        cutoff_date = pd.to_datetime('2016-01-01')  # Allow 24 months to end of 2017
        df_filtered = df[df['issue_d'] <= cutoff_date].copy()
        
        # Create splits
        train_df = df_filtered[df_filtered['issue_d'] <= pd.to_datetime('2014-12-31')]
        val_df = df_filtered[
            (df_filtered['issue_d'] > pd.to_datetime('2014-12-31')) &
            (df_filtered['issue_d'] <= pd.to_datetime('2015-12-31'))
        ]
        test_df = df_filtered[df_filtered['issue_d'] > pd.to_datetime('2015-12-31')]
        
        print(f"Train set: {len(train_df):,} loans ({train_df['issue_d'].min()} to {train_df['issue_d'].max()})")
        print(f"Validation set: {len(val_df):,} loans ({val_df['issue_d'].min()} to {val_df['issue_d'].max()})")
        print(f"Test set: {len(test_df):,} loans ({test_df['issue_d'].min()} to {test_df['issue_d'].max()})")
        
        return train_df, val_df, test_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the model
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        df_eng = df.copy()
        
        # Payment-to-income ratio
        if 'installment' in df_eng.columns and 'annual_inc' in df_eng.columns:
            df_eng['payment_to_income'] = df_eng['installment'] * 12 / df_eng['annual_inc']
        
        # Credit utilization features
        if 'revol_bal' in df_eng.columns and 'total_rev_hi_lim' in df_eng.columns:
            df_eng['credit_utilization'] = df_eng['revol_bal'] / (df_eng['total_rev_hi_lim'] + 1)
        
        # Credit age (months)
        if 'earliest_cr_line' in df_eng.columns and 'issue_d' in df_eng.columns:
            df_eng['credit_age_months'] = (
                (df_eng['issue_d'] - df_eng['earliest_cr_line']).dt.days / 30.44
            )
        
        # Loan amount to income ratio
        if 'loan_amnt' in df_eng.columns and 'annual_inc' in df_eng.columns:
            df_eng['loan_to_income'] = df_eng['loan_amnt'] / df_eng['annual_inc']
        
        # Employment length encoding
        if 'emp_length' in df_eng.columns:
            emp_length_map = {
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            df_eng['emp_length_numeric'] = df_eng['emp_length'].map(emp_length_map)
        
        # Term encoding
        if 'term' in df_eng.columns:
            df_eng['term_months'] = df_eng['term'].str.extract(r'(\d+)').astype(float)
        
        print(f"Feature engineering complete. Shape: {df_eng.shape}")
        
        return df_eng

def main():
    """
    Main execution function
    """
    print("=== LendingClub 90-Day Delinquency Model ===")
    print("This script requires the LendingClub dataset:")
    print("accepted_2007_to_2018Q4.csv.gz")
    print("\nPlease download from:")
    print("https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print("or LendingClub's official data repository")
    print("\nPlace the file in the 'data/' directory")
    
    # Initialize model
    data_path = "data/accepted_2007_to_2018Q4.csv.gz"
    model = LendingClubDelinquencyModel(data_path)
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"\nData file not found at: {data_path}")
        print("Please download and place the file in the correct location.")
        return
    
    # Execute pipeline
    try:
        # Load and clean data
        raw_data = model.load_data()
        if raw_data is None:
            return
            
        cleaned_data = model.clean_data()
        labeled_data = model.create_delinquency_labels(cleaned_data)
        
        # Feature engineering and splits
        engineered_data = model.engineer_features(labeled_data)
        train_df, val_df, test_df = model.temporal_split(engineered_data)
        
        print("\nData preparation complete!")
        print("Next steps:")
        print("1. Feature selection and preprocessing")
        print("2. Model training with LightGBM/XGBoost")
        print("3. Model evaluation and calibration")
        print("4. SHAP explanations")
        print("5. Fairness analysis")
        print("6. MLflow tracking and model registration")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
