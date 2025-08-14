"""
Enhanced LendingClub Pipeline Demo
=================================

This script demonstrates the enhanced pipeline capabilities using the provided
data files: train/test datasets, data dictionary, and pre-trained model.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Pandas not available. Please install dependencies first.")

def check_provided_files():
    """Check if all provided files are available"""
    print("=" * 60)
    print("CHECKING PROVIDED DATA FILES")
    print("=" * 60)
    
    required_files = {
        'data/train_lending_club.csv': 'Training dataset',
        'data/test_lending_club.csv': 'Test dataset', 
        'data/LCDataDictionary.csv': 'Data dictionary',
        'data/model.joblib': 'Pre-trained model'
    }
    
    all_present = True
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            print(f"[OK] {description}: {file_path} ({file_size:.1f} MB)")
        else:
            print(f"[MISSING] {description}: {file_path}")
            all_present = False
    
    return all_present

def demonstrate_data_loading():
    """Demonstrate loading and exploring the provided datasets"""
    if not PANDAS_AVAILABLE:
        print("Pandas required for data loading demonstration")
        return
    
    print("\n" + "=" * 60)
    print("DATA LOADING DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Load datasets
        print("Loading training data...")
        train_df = pd.read_csv('data/train_lending_club.csv')
        
        print("Loading test data...")
        test_df = pd.read_csv('data/test_lending_club.csv')
        
        print("Loading data dictionary...")
        data_dict = pd.read_csv('data/LCDataDictionary.csv')
        
        # Display basic information
        print(f"\nDATASET OVERVIEW:")
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Test samples: {len(test_df):,}")
        print(f"   Total features: {len(train_df.columns) - 1}")  # Exclude target
        print(f"   Feature definitions: {len(data_dict)}")
        
        # Target distribution
        if 'loan_status' in train_df.columns:
            train_pos_rate = train_df['loan_status'].mean()
            test_pos_rate = test_df['loan_status'].mean()
            print(f"   Train positive rate: {train_pos_rate:.3f}")
            print(f"   Test positive rate: {test_pos_rate:.3f}")
        
        # Feature types
        numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nFEATURE TYPES:")
        print(f"   Numerical features: {len(numerical_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        
        # Sample features
        print(f"\nSAMPLE FEATURES:")
        print("   Numerical:", numerical_features[:5])
        print("   Categorical:", categorical_features[:5])
        
        # Missing value analysis
        missing_analysis = train_df.isnull().sum()
        features_with_missing = missing_analysis[missing_analysis > 0]
        
        print(f"\nMISSING VALUES:")
        if len(features_with_missing) > 0:
            print(f"   Features with missing values: {len(features_with_missing)}")
            for feature, count in features_with_missing.head(5).items():
                pct = (count / len(train_df)) * 100
                print(f"     {feature}: {count:,} ({pct:.1f}%)")
        else:
            print("   No missing values detected")
        
        # Data dictionary sample
        print(f"\nDATA DICTIONARY SAMPLE:")
        if len(data_dict) > 0:
            # Rename columns for consistency
            if 'LoanStatNew' in data_dict.columns:
                data_dict = data_dict.rename(columns={
                    'LoanStatNew': 'feature_name',
                    'Description': 'description'
                })
            
            for _, row in data_dict.head(3).iterrows():
                feature_name = row.iloc[0]  # First column
                description = row.iloc[1]   # Second column
                print(f"   {feature_name}: {description}")
        
        return train_df, test_df, data_dict
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        return None, None, None

def demonstrate_model_evaluation():
    """Demonstrate evaluation of the pre-trained model"""
    if not PANDAS_AVAILABLE:
        print("Pandas required for model evaluation")
        return
    
    print("\n" + "=" * 60)
    print("PRE-TRAINED MODEL EVALUATION")
    print("=" * 60)
    
    try:
        # Import the enhanced data loader
        from data_loader import LendingClubDataLoader
        
        # Initialize loader
        loader = LendingClubDataLoader()
        
        # Load data and model
        train_df, test_df = loader.load_train_test_data()
        model = loader.load_pretrained_model()
        
        if model is None:
            print("[ERROR] Pre-trained model not available")
            return
        
        # Prepare test data
        X_test, y_test = loader.prepare_features(test_df)
        
        # Evaluate model
        print("Evaluating pre-trained model on test data...")
        metrics = loader.evaluate_pretrained_model(X_test, y_test)
        
        # Get feature importance
        importance_df = loader.get_model_feature_importance()
        
        if importance_df is not None:
            print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                feature_desc = loader.get_feature_description(row['feature'])
                print(f"   {i:2d}. {row['feature']}: {row['importance']:.4f}")
                if feature_desc != "Description not found":
                    desc_short = feature_desc[:60] + "..." if len(feature_desc) > 60 else feature_desc
                    print(f"       -> {desc_short}")
        
        # Risk factor analysis
        if importance_df is not None:
            print(f"\nKEY RISK INDICATORS:")
            
            # Identify risk-related features
            risk_keywords = ['delinq', 'dti', 'revol_util', 'pub_rec', 'inq', 'fico']
            risk_features = []
            
            for _, row in importance_df.iterrows():
                feature_name = row['feature'].lower()
                if any(keyword in feature_name for keyword in risk_keywords):
                    risk_features.append((row['feature'], row['importance']))
            
            for feature, importance in risk_features[:5]:
                print(f"   - {feature}: {importance:.4f}")
        
        return metrics, importance_df
        
    except Exception as e:
        print(f"[ERROR] Error evaluating model: {e}")
        return None, None

def demonstrate_enhanced_features():
    """Demonstrate enhanced pipeline features"""
    print("\n" + "=" * 60)
    print("ENHANCED PIPELINE FEATURES")
    print("=" * 60)
    
    features = [
        "[1] Automated Data Loading",
        "   - Pre-split train/test datasets",
        "   - Integrated data dictionary lookup",
        "   - Pre-trained model evaluation",
        "",
        "[2] Enhanced Model Analysis", 
        "   - Feature importance with descriptions",
        "   - Risk factor identification",
        "   - Performance benchmarking",
        "",
        "[3] Advanced Fairness Analysis",
        "   - Geographic grouping (state-level)",
        "   - Income quartile analysis", 
        "   - FICO score group comparisons",
        "",
        "[4] Comprehensive Interpretability",
        "   - SHAP explanations for pre-trained model",
        "   - Feature interaction analysis",
        "   - Counterfactual recommendations",
        "",
        "[5] MLOps Integration",
        "   - Enhanced model cards with data dictionary",
        "   - Automated governance reporting",
        "   - Performance monitoring setup"
    ]
    
    for feature in features:
        print(feature)

def demonstrate_usage_examples():
    """Show practical usage examples"""
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    
    examples = [
        "1. Quick Model Evaluation:",
        "   python enhanced_demo.py",
        "",
        "2. Full Enhanced Pipeline:",
        "   python main_pipeline.py",
        "   # Automatically detects provided data files",
        "",
        "3. Interactive Data Exploration:",
        "   from data_loader import LendingClubDataLoader",
        "   loader = LendingClubDataLoader()",
        "   train_df, test_df = loader.load_train_test_data()",
        "",
        "4. Feature Analysis:",
        "   feature_summary = loader.create_feature_summary()",
        "   print(feature_summary.head())",
        "",
        "5. Model Interpretability:",
        "   from interpretability import ModelInterpreter", 
        "   interpreter = ModelInterpreter(model, feature_names)",
        "   interpreter.setup_shap_explainer(background_data)"
    ]
    
    for example in examples:
        print(example)

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("ENHANCED LENDINGCLUB PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Check if provided files are available
    files_available = check_provided_files()
    
    if not files_available:
        print(f"\n[ERROR] Some required files are missing.")
        print("Please ensure all data files are in the data/ directory:")
        print("  - train_lending_club.csv")
        print("  - test_lending_club.csv") 
        print("  - LCDataDictionary.csv")
        print("  - model.joblib")
        return False
    
    print(f"\n[OK] All required files found!")
    
    # Demonstrate data loading
    train_df, test_df, data_dict = demonstrate_data_loading()
    
    # Demonstrate model evaluation
    if train_df is not None:
        result = demonstrate_model_evaluation()
        if result is not None:
            metrics, importance_df = result
        else:
            metrics, importance_df = None, None
    
    # Show enhanced features
    demonstrate_enhanced_features()
    
    # Show usage examples
    demonstrate_usage_examples()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    next_steps = [
        "Ready to run enhanced pipeline:",
        "",
        "1. Install dependencies (if not already done):",
        "   pip install -r requirements.txt",
        "",
        "2. Run the enhanced pipeline:",
        "   python main_pipeline.py",
        "",
        "3. The pipeline will automatically:",
        "   - Detect your provided data files",
        "   - Load and evaluate the pre-trained model", 
        "   - Generate comprehensive analysis reports",
        "   - Create enhanced visualizations",
        "   - Log everything to MLflow",
        "",
        "4. Explore individual components:",
        "   - Data loading: python data_loader.py",
        "   - Model analysis: python -c 'from data_loader import *; main()'",
        "",
        "Your data is ready for advanced ML analysis!"
    ]
    
    for step in next_steps:
        print(step)
    
    print("\n" + "=" * 60)
    print("ENHANCED DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    main()
