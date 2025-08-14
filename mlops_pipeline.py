"""
MLOps Pipeline for LendingClub 90-Day Delinquency Prediction
==========================================================

This module handles model tracking, registration, monitoring, and governance
using MLflow and other MLOps best practices.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

class MLOpsManager:
    """
    Manages MLOps lifecycle for the delinquency prediction model
    """
    
    def __init__(self, experiment_name: str = "lending_club_delinquency"):
        """
        Initialize MLOps manager
        
        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.experiment_id = None
        self.run_id = None
        self.model_uri = None
        
        # Set up MLflow
        self.setup_mlflow()
    
    def setup_mlflow(self):
        """
        Setup MLflow experiment and tracking
        """
        print(f"Setting up MLflow experiment: {self.experiment_name}")
        
        # Set tracking URI (local for this example)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Error setting up experiment: {e}")
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        
        mlflow.set_experiment(self.experiment_name)
        print(f"Experiment ID: {self.experiment_id}")
    
    def log_model_training(self, model, model_type: str, train_data: Dict,
                          val_data: Dict, test_data: Dict, metrics: Dict,
                          feature_names: List[str], hyperparams: Dict = None):
        """
        Log model training run to MLflow
        
        Args:
            model: Trained model
            model_type: Type of model ('lightgbm', 'xgboost', etc.)
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            metrics: Model performance metrics
            feature_names: List of feature names
            hyperparams: Model hyperparameters
        """
        print(f"Logging {model_type} model training to MLflow...")
        
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            
            # Log parameters
            if hyperparams:
                mlflow.log_params(hyperparams)
            
            # Log dataset info
            mlflow.log_param("train_size", len(train_data['X']))
            mlflow.log_param("val_size", len(val_data['X']))
            mlflow.log_param("test_size", len(test_data['X']))
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("model_type", model_type)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Save and log feature importance
                importance_path = "feature_importance.csv"
                feature_importance.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
            
            # Create model signature
            signature = infer_signature(train_data['X'], model.predict_proba(train_data['X']))
            
            # Log model based on type
            if model_type == 'lightgbm':
                mlflow.lightgbm.log_model(
                    model, "model", signature=signature,
                    input_example=train_data['X'].head(5)
                )
            elif model_type == 'xgboost':
                mlflow.xgboost.log_model(
                    model, "model", signature=signature,
                    input_example=train_data['X'].head(5)
                )
            else:
                mlflow.sklearn.log_model(
                    model, "model", signature=signature,
                    input_example=train_data['X'].head(5)
                )
            
            # Log additional artifacts
            self.log_training_artifacts(model, test_data, feature_names)
            
            self.model_uri = f"runs:/{self.run_id}/model"
            print(f"Model logged with run ID: {self.run_id}")
    
    def log_training_artifacts(self, model, test_data: Dict, feature_names: List[str]):
        """
        Log additional training artifacts
        
        Args:
            model: Trained model
            test_data: Test data
            feature_names: Feature names
        """
        # Create and log ROC curve
        y_pred_proba = model.predict_proba(test_data['X'])[:, 1]
        
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(test_data['y'], y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(test_data["y"], y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('roc_curve.png')
        plt.close()
        os.remove('roc_curve.png')
        
        # Create and log Precision-Recall curve
        precision, recall, _ = precision_recall_curve(test_data['y'], y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('pr_curve.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('pr_curve.png')
        plt.close()
        os.remove('pr_curve.png')
    
    def register_model(self, model_name: str = "lending_club_delinquency_model",
                      description: str = None) -> str:
        """
        Register model in MLflow Model Registry
        
        Args:
            model_name: Name for registered model
            description: Model description
            
        Returns:
            Model version
        """
        if not self.model_uri:
            raise ValueError("No model URI available. Train and log a model first.")
        
        print(f"Registering model: {model_name}")
        
        if description is None:
            description = f"LendingClub 90-day delinquency prediction model trained on {datetime.now().strftime('%Y-%m-%d')}"
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=self.model_uri,
            name=model_name,
            description=description
        )
        
        print(f"Model registered as version {model_version.version}")
        return model_version.version
    
    def create_model_card(self, model_name: str, model_version: str,
                         metrics: Dict, feature_names: List[str],
                         fairness_results: Dict = None) -> Dict[str, Any]:
        """
        Create comprehensive model card
        
        Args:
            model_name: Name of the model
            model_version: Model version
            metrics: Performance metrics
            feature_names: List of features
            fairness_results: Fairness analysis results
            
        Returns:
            Model card dictionary
        """
        print("Creating model card...")
        
        model_card = {
            "model_details": {
                "name": model_name,
                "version": model_version,
                "date": datetime.now().isoformat(),
                "model_type": "Gradient Boosting (LightGBM/XGBoost)",
                "task": "Binary Classification - 90+ Day Delinquency Prediction",
                "framework": "scikit-learn, LightGBM, XGBoost"
            },
            "intended_use": {
                "primary_use": "Predict probability of loan becoming 90+ days delinquent",
                "intended_users": "Credit risk analysts, loan underwriters",
                "out_of_scope": "Not for use in final credit decisions without human review"
            },
            "training_data": {
                "source": "LendingClub 2007-2018 accepted loans",
                "size": "1.6M+ loans",
                "temporal_split": "Train: 2010-2014, Val: 2015, Test: 2016-2017",
                "features": len(feature_names),
                "target_definition": "Late(31-120), Default, or Charged Off within 24 months"
            },
            "performance": {
                "test_metrics": metrics,
                "evaluation_approach": "Temporal validation with 24-month observation window"
            },
            "ethical_considerations": {
                "fairness_analysis": fairness_results is not None,
                "bias_mitigation": "Monotonic constraints, fairness monitoring",
                "limitations": [
                    "Historical data may not reflect current economic conditions",
                    "Model performance may vary across demographic groups",
                    "Requires regular monitoring and retraining"
                ]
            },
            "technical_details": {
                "features": feature_names[:20],  # Top 20 features
                "preprocessing": [
                    "Missing value imputation",
                    "Categorical encoding",
                    "Feature engineering (DTI, utilization ratios)"
                ],
                "model_constraints": "Monotonic constraints on key risk features"
            }
        }
        
        if fairness_results:
            model_card["fairness_metrics"] = {
                "max_tpr_gap": fairness_results.get('summary', {}).get('max_tpr_gap', 'N/A'),
                "max_fpr_gap": fairness_results.get('summary', {}).get('max_fpr_gap', 'N/A'),
                "groups_analyzed": list(fairness_results.get('detailed_results', {}).keys())
            }
        
        # Save model card
        model_card_path = f"model_card_{model_name}_v{model_version}.json"
        with open(model_card_path, 'w') as f:
            json.dump(model_card, f, indent=2)
        
        # Log to MLflow
        with mlflow.start_run(run_id=self.run_id):
            mlflow.log_artifact(model_card_path)
        
        os.remove(model_card_path)
        
        return model_card
    
    def compute_drift_metrics(self, reference_data: pd.DataFrame,
                            current_data: pd.DataFrame,
                            feature_names: List[str]) -> Dict[str, float]:
        """
        Compute Population Stability Index (PSI) for drift detection
        
        Args:
            reference_data: Reference dataset (e.g., training data)
            current_data: Current dataset to compare
            feature_names: Features to analyze
            
        Returns:
            Dictionary of PSI values by feature
        """
        print("Computing drift metrics (PSI)...")
        
        psi_values = {}
        
        for feature in feature_names:
            if feature not in reference_data.columns or feature not in current_data.columns:
                continue
            
            try:
                psi = self.calculate_psi(
                    reference_data[feature], 
                    current_data[feature]
                )
                psi_values[feature] = psi
            except Exception as e:
                print(f"Error calculating PSI for {feature}: {e}")
                psi_values[feature] = np.nan
        
        return psi_values
    
    def calculate_psi(self, reference: pd.Series, current: pd.Series,
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        Args:
            reference: Reference distribution
            current: Current distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Remove missing values
        ref_clean = reference.dropna()
        cur_clean = current.dropna()
        
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return np.nan
        
        # Create bins based on reference data
        if ref_clean.dtype in ['object', 'category']:
            # Categorical feature
            ref_counts = ref_clean.value_counts(normalize=True)
            cur_counts = cur_clean.value_counts(normalize=True)
            
            # Align categories
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_probs = [ref_counts.get(cat, 1e-6) for cat in all_categories]
            cur_probs = [cur_counts.get(cat, 1e-6) for cat in all_categories]
            
        else:
            # Numerical feature
            try:
                _, bin_edges = np.histogram(ref_clean, bins=bins)
                ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
                cur_counts, _ = np.histogram(cur_clean, bins=bin_edges)
                
                # Convert to probabilities
                ref_probs = ref_counts / ref_counts.sum()
                cur_probs = cur_counts / cur_counts.sum()
                
                # Add small epsilon to avoid log(0)
                ref_probs = np.where(ref_probs == 0, 1e-6, ref_probs)
                cur_probs = np.where(cur_probs == 0, 1e-6, cur_probs)
                
            except:
                return np.nan
        
        # Calculate PSI
        psi = np.sum((np.array(cur_probs) - np.array(ref_probs)) * 
                     np.log(np.array(cur_probs) / np.array(ref_probs)))
        
        return psi
    
    def monitor_model_performance(self, model, current_data: Dict,
                                reference_metrics: Dict,
                                feature_names: List[str]) -> Dict[str, Any]:
        """
        Monitor model performance and detect degradation
        
        Args:
            model: Trained model
            current_data: Current data with X and y
            reference_metrics: Reference performance metrics
            feature_names: Feature names
            
        Returns:
            Monitoring report
        """
        print("Monitoring model performance...")
        
        # Calculate current performance
        y_pred_proba = model.predict_proba(current_data['X'])[:, 1]
        current_auc = roc_auc_score(current_data['y'], y_pred_proba)
        
        precision, recall, _ = precision_recall_curve(current_data['y'], y_pred_proba)
        current_pr_auc = auc(recall, precision)
        
        # Compare with reference
        auc_drift = abs(current_auc - reference_metrics.get('roc_auc', current_auc))
        pr_auc_drift = abs(current_pr_auc - reference_metrics.get('pr_auc', current_pr_auc))
        
        # Drift thresholds
        auc_threshold = 0.05  # 5% degradation threshold
        pr_auc_threshold = 0.05
        
        monitoring_report = {
            "timestamp": datetime.now().isoformat(),
            "current_performance": {
                "roc_auc": current_auc,
                "pr_auc": current_pr_auc
            },
            "reference_performance": reference_metrics,
            "performance_drift": {
                "auc_drift": auc_drift,
                "pr_auc_drift": pr_auc_drift
            },
            "alerts": {
                "auc_degradation": auc_drift > auc_threshold,
                "pr_auc_degradation": pr_auc_drift > pr_auc_threshold
            },
            "recommendations": []
        }
        
        # Generate recommendations
        if monitoring_report["alerts"]["auc_degradation"]:
            monitoring_report["recommendations"].append(
                f"AUC degradation detected ({auc_drift:.3f}). Consider model retraining."
            )
        
        if monitoring_report["alerts"]["pr_auc_degradation"]:
            monitoring_report["recommendations"].append(
                f"PR-AUC degradation detected ({pr_auc_drift:.3f}). Review model performance."
            )
        
        if not any(monitoring_report["alerts"].values()):
            monitoring_report["recommendations"].append(
                "Model performance is stable. Continue monitoring."
            )
        
        return monitoring_report
    
    def create_governance_report(self, model_card: Dict, fairness_results: Dict,
                               drift_metrics: Dict, monitoring_report: Dict) -> Dict[str, Any]:
        """
        Create comprehensive governance report
        
        Args:
            model_card: Model card information
            fairness_results: Fairness analysis results
            drift_metrics: Data drift metrics
            monitoring_report: Performance monitoring report
            
        Returns:
            Governance report
        """
        print("Creating governance report...")
        
        governance_report = {
            "report_date": datetime.now().isoformat(),
            "model_info": {
                "name": model_card["model_details"]["name"],
                "version": model_card["model_details"]["version"],
                "deployment_date": model_card["model_details"]["date"]
            },
            "performance_summary": {
                "current_metrics": monitoring_report["current_performance"],
                "performance_stable": not any(monitoring_report["alerts"].values())
            },
            "fairness_summary": {
                "analysis_completed": fairness_results is not None,
                "max_disparity": fairness_results.get('summary', {}).get('max_tpr_gap', 'N/A') if fairness_results else 'N/A',
                "groups_analyzed": len(fairness_results.get('detailed_results', {})) if fairness_results else 0
            },
            "data_quality": {
                "drift_detected": any(psi > 0.2 for psi in drift_metrics.values() if not np.isnan(psi)),
                "high_drift_features": [
                    feature for feature, psi in drift_metrics.items() 
                    if not np.isnan(psi) and psi > 0.2
                ],
                "avg_psi": np.nanmean(list(drift_metrics.values()))
            },
            "compliance_status": {
                "model_documented": True,
                "fairness_assessed": fairness_results is not None,
                "monitoring_active": True,
                "explainability_available": True
            },
            "action_items": []
        }
        
        # Generate action items
        if governance_report["data_quality"]["drift_detected"]:
            governance_report["action_items"].append(
                "High data drift detected. Review feature distributions and consider retraining."
            )
        
        if not governance_report["performance_summary"]["performance_stable"]:
            governance_report["action_items"].append(
                "Performance degradation detected. Investigate causes and retrain if necessary."
            )
        
        if fairness_results and len(fairness_results.get('summary', {}).get('problematic_groups', [])) > 0:
            governance_report["action_items"].append(
                "Fairness issues identified. Review model decisions for affected groups."
            )
        
        if not governance_report["action_items"]:
            governance_report["action_items"].append(
                "No immediate action required. Continue regular monitoring."
            )
        
        return governance_report
