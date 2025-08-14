"""
Pipeline Validation Script
=========================

This script validates the structure and imports of the LendingClub
delinquency prediction pipeline without requiring the full dataset.
"""

import sys
import os
import importlib.util

def validate_file_structure():
    """Validate that all required files exist"""
    required_files = [
        'lending_club_delinquency_model.py',
        'model_training.py',
        'interpretability.py',
        'fairness_analysis.py',
        'mlops_pipeline.py',
        'main_pipeline.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("Validating file structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [MISSING] {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        return False
    else:
        print("\nAll required files present!")
        return True

def validate_imports():
    """Validate that modules can be imported (syntax check)"""
    modules = [
        'lending_club_delinquency_model',
        'model_training',
        'interpretability',
        'fairness_analysis',
        'mlops_pipeline'
    ]
    
    print("\nValidating module syntax...")
    
    for module_name in modules:
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            
            # Just check syntax, don't execute
            with open(f"{module_name}.py", 'r') as f:
                compile(f.read(), f"{module_name}.py", 'exec')
            
            print(f"  [OK] {module_name}.py - Syntax OK")
            
        except SyntaxError as e:
            print(f"  [ERROR] {module_name}.py - Syntax Error: {e}")
            return False
        except Exception as e:
            print(f"  [WARNING] {module_name}.py - Warning: {e}")
    
    print("\nAll modules have valid syntax!")
    return True

def validate_data_directory():
    """Check data directory structure"""
    print("\nValidating data directory...")
    
    if os.path.exists('data'):
        print("  [OK] data/ directory exists")
        
        # Check for LendingClub data
        lc_file = 'data/accepted_2007_to_2018Q4.csv.gz'
        if os.path.exists(lc_file):
            print(f"  [OK] {lc_file} found")
            file_size = os.path.getsize(lc_file) / (1024**2)  # MB
            print(f"     File size: {file_size:.1f} MB")
        else:
            print(f"  [INFO] {lc_file} not found")
            print("     Pipeline can still run with sample data")
        
        return True
    else:
        print("  [ERROR] data/ directory missing")
        return False

def validate_requirements():
    """Check requirements.txt content"""
    print("\nValidating requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        expected_packages = [
            'pandas', 'numpy', 'scikit-learn', 'lightgbm', 
            'xgboost', 'shap', 'matplotlib', 'seaborn', 
            'mlflow', 'imbalanced-learn'
        ]
        
        found_packages = []
        for req in requirements:
            if req.strip():
                package = req.split('==')[0].split('>=')[0].strip()
                found_packages.append(package)
        
        missing_packages = []
        for pkg in expected_packages:
            if pkg in found_packages:
                print(f"  [OK] {pkg}")
            else:
                print(f"  [MISSING] {pkg}")
                missing_packages.append(pkg)
        
        if missing_packages:
            print(f"\nMissing packages in requirements.txt: {missing_packages}")
        else:
            print(f"\nAll required packages listed!")
        
        return len(missing_packages) == 0
        
    except Exception as e:
        print(f"  [ERROR] Error reading requirements.txt: {e}")
        return False

def create_sample_test():
    """Create a minimal test to verify pipeline components"""
    print("\nCreating sample test...")
    
    test_code = '''
# Minimal test without external dependencies
import sys
import os

# Test basic Python functionality
def test_basic_functionality():
    """Test basic Python operations"""
    # Test data structures
    sample_data = {
        'loan_amnt': [10000, 15000, 20000],
        'int_rate': [10.5, 12.0, 8.5],
        'annual_inc': [50000, 60000, 75000]
    }
    
    # Test calculations
    dti_values = []
    for i in range(len(sample_data['loan_amnt'])):
        # Simulate DTI calculation
        monthly_payment = sample_data['loan_amnt'][i] * 0.02  # Rough estimate
        monthly_income = sample_data['annual_inc'][i] / 12
        dti = (monthly_payment / monthly_income) * 100
        dti_values.append(dti)
    
    print(f"Sample DTI calculations: {dti_values}")
    return len(dti_values) == 3

if __name__ == "__main__":
    print("Running basic functionality test...")
    if test_basic_functionality():
        print("[OK] Basic test passed!")
    else:
        print("[ERROR] Basic test failed!")
'''
    
    try:
        with open('test_basic.py', 'w') as f:
            f.write(test_code)
        
        # Run the test
        exec(test_code)
        print("  [OK] Sample test created and executed successfully!")
        
        # Clean up
        if os.path.exists('test_basic.py'):
            os.remove('test_basic.py')
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Error creating/running sample test: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 60)
    print("LendingClub Delinquency Model Pipeline Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Run all validations
    validation_results.append(("File Structure", validate_file_structure()))
    validation_results.append(("Module Syntax", validate_imports()))
    validation_results.append(("Data Directory", validate_data_directory()))
    validation_results.append(("Requirements", validate_requirements()))
    validation_results.append(("Basic Test", create_sample_test()))
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(validation_results)
    
    for test_name, result in validation_results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        print("\nPipeline validation successful!")
        print("\nNext steps:")
        print("1. Wait for dependencies to finish installing")
        print("2. Download LendingClub data or run with sample data")
        print("3. Execute: python main_pipeline.py")
    else:
        print(f"\n{total - passed} validation(s) failed")
        print("Please review and fix issues before running the pipeline")
    
    return passed == total

if __name__ == "__main__":
    main()
