"""
Model Interpretability Module for LendingClub 90-Day Delinquency Prediction
==========================================================================

This module provides comprehensive model interpretability using SHAP,
feature importance analysis, and counterfactual explanations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

class ModelInterpreter:
    """
    Handles model interpretability and explanations
    """
    
    def __init__(self, model, feature_names: List[str], 
                 categorical_features: List[str] = None):
        """
        Initialize interpreter
        
        Args:
            model: Trained model
            feature_names: List of feature names
            categorical_features: List of categorical feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.explainer = None
        self.shap_values = None
        
    def setup_shap_explainer(self, background_data: pd.DataFrame, 
                           explainer_type: str = 'tree'):
        """
        Setup SHAP explainer
        
        Args:
            background_data: Background dataset for SHAP
            explainer_type: Type of explainer ('tree', 'kernel', 'linear')
        """
        print(f"Setting up SHAP {explainer_type} explainer...")
        
        if explainer_type == 'tree':
            # For tree-based models (LightGBM, XGBoost)
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # Model-agnostic explainer
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                background_data.sample(min(100, len(background_data)))
            )
        else:
            raise ValueError(f"Unsupported explainer type: {explainer_type}")
    
    def compute_shap_values(self, data: pd.DataFrame, 
                          sample_size: int = None) -> np.ndarray:
        """
        Compute SHAP values for the dataset
        
        Args:
            data: Input data
            sample_size: Optional sample size for large datasets
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer not initialized. Call setup_shap_explainer first.")
        
        # Sample data if too large
        if sample_size and len(data) > sample_size:
            data_sample = data.sample(sample_size, random_state=42)
        else:
            data_sample = data
        
        print(f"Computing SHAP values for {len(data_sample)} samples...")
        
        # Compute SHAP values
        if hasattr(self.explainer, 'shap_values'):
            # For tree explainers
            shap_values = self.explainer.shap_values(data_sample)
            # Handle binary classification case
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
        else:
            # For other explainers
            shap_values = self.explainer(data_sample)
            if hasattr(shap_values, 'values'):
                shap_values = shap_values.values
        
        self.shap_values = shap_values
        return shap_values
    
    def plot_feature_importance(self, importance_type: str = 'gain', 
                              top_k: int = 20):
        """
        Plot feature importance
        
        Args:
            importance_type: Type of importance ('gain', 'split', 'permutation')
            top_k: Number of top features to show
        """
        print(f"Plotting {importance_type} feature importance...")
        
        if hasattr(self.model, 'feature_importances_'):
            # For sklearn-compatible models
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
        elif hasattr(self.model, 'get_score'):
            # For XGBoost
            importance_dict = self.model.get_score(importance_type=importance_type)
            feature_imp = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in importance_dict.items()
            ]).sort_values('importance', ascending=False)
            
        else:
            print("Model does not support feature importance extraction")
            return
        
        # Plot top features
        plt.figure(figsize=(10, 8))
        top_features = feature_imp.head(top_k)
        
        sns.barplot(data=top_features, y='feature', x='importance', 
                   palette='viridis')
        plt.title(f'Top {top_k} Feature Importance ({importance_type})')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
        
        return feature_imp
    
    def plot_shap_summary(self, data: pd.DataFrame = None, 
                         plot_type: str = 'dot', max_display: int = 20):
        """
        Plot SHAP summary plot
        
        Args:
            data: Input data for plotting
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("SHAP values not computed. Call compute_shap_values first.")
            return
        
        print(f"Creating SHAP {plot_type} summary plot...")
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'dot':
            shap.summary_plot(
                self.shap_values, data, 
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        elif plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        elif plot_type == 'violin':
            shap.summary_plot(
                self.shap_values, data,
                feature_names=self.feature_names,
                plot_type='violin',
                max_display=max_display,
                show=False
            )
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, instance_idx: int, data: pd.DataFrame,
                           max_display: int = 10):
        """
        Plot SHAP waterfall for individual prediction
        
        Args:
            instance_idx: Index of instance to explain
            data: Input data
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            print("SHAP values not computed. Call compute_shap_values first.")
            return
        
        print(f"Creating SHAP waterfall for instance {instance_idx}...")
        
        # Get prediction
        pred_proba = self.model.predict_proba(data.iloc[[instance_idx]])[0, 1]
        
        plt.figure(figsize=(10, 8))
        
        # Create explanation object for waterfall
        if hasattr(shap, 'Explanation'):
            explanation = shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                data=data.iloc[instance_idx].values,
                feature_names=self.feature_names
            )
            shap.waterfall_plot(explanation, max_display=max_display, show=False)
        else:
            # Fallback for older SHAP versions
            shap.force_plot(
                self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0,
                self.shap_values[instance_idx],
                data.iloc[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
        
        plt.title(f'SHAP Explanation - Instance {instance_idx}\n'
                 f'Predicted Probability: {pred_proba:.3f}')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_dependence(self, feature_name: str, data: pd.DataFrame,
                           interaction_feature: str = None):
        """
        Plot SHAP dependence plot for a feature
        
        Args:
            feature_name: Feature to analyze
            data: Input data
            interaction_feature: Optional interaction feature
        """
        if self.shap_values is None:
            print("SHAP values not computed. Call compute_shap_values first.")
            return
        
        if feature_name not in self.feature_names:
            print(f"Feature '{feature_name}' not found in feature names")
            return
        
        feature_idx = self.feature_names.index(feature_name)
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, self.shap_values, data,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.tight_layout()
        plt.show()
    
    def generate_counterfactuals(self, instance: pd.Series, 
                               target_change: float = -0.1,
                               max_changes: int = 3) -> Dict[str, Any]:
        """
        Generate counterfactual explanations
        
        Args:
            instance: Single instance to explain
            target_change: Desired change in prediction probability
            max_changes: Maximum number of feature changes
            
        Returns:
            Dictionary with counterfactual suggestions
        """
        print("Generating counterfactual explanations...")
        
        # Get current prediction
        current_pred = self.model.predict_proba(instance.values.reshape(1, -1))[0, 1]
        target_pred = current_pred + target_change
        
        # Get SHAP values for this instance
        if self.shap_values is None:
            instance_shap = self.explainer.shap_values(instance.values.reshape(1, -1))
            if isinstance(instance_shap, list):
                instance_shap = instance_shap[1][0]  # Positive class
            else:
                instance_shap = instance_shap[0]
        else:
            # Use pre-computed SHAP values if available
            instance_shap = self.shap_values[0] if len(self.shap_values.shape) > 1 else self.shap_values
        
        # Identify features with highest impact
        feature_impacts = pd.DataFrame({
            'feature': self.feature_names,
            'shap_value': instance_shap,
            'current_value': instance.values
        })
        
        # Sort by absolute SHAP value (highest impact first)
        feature_impacts['abs_shap'] = np.abs(feature_impacts['shap_value'])
        feature_impacts = feature_impacts.sort_values('abs_shap', ascending=False)
        
        suggestions = []
        
        # Generate suggestions for top impactful features
        for _, row in feature_impacts.head(max_changes * 2).iterrows():
            feature = row['feature']
            current_val = row['current_value']
            shap_val = row['shap_value']
            
            # Skip categorical features for now
            if feature in self.categorical_features:
                continue
            
            # Suggest changes that would improve the prediction
            if shap_val > 0:  # Feature increases risk
                # Suggest decreasing the feature
                if current_val > 0:
                    suggested_val = current_val * 0.8  # 20% decrease
                    change_direction = "decrease"
                else:
                    continue
            else:  # Feature decreases risk
                # Suggest increasing the feature
                suggested_val = current_val * 1.2  # 20% increase
                change_direction = "increase"
            
            # Create suggestion
            suggestion = {
                'feature': feature,
                'current_value': current_val,
                'suggested_value': suggested_val,
                'change_direction': change_direction,
                'shap_impact': shap_val,
                'change_magnitude': abs(suggested_val - current_val)
            }
            
            suggestions.append(suggestion)
            
            if len(suggestions) >= max_changes:
                break
        
        # Create actionable recommendations
        recommendations = []
        for sugg in suggestions:
            if sugg['feature'] == 'dti':
                rec = f"Reduce debt-to-income ratio from {sugg['current_value']:.2f} to {sugg['suggested_value']:.2f}"
            elif sugg['feature'] == 'revol_util':
                rec = f"Reduce credit utilization from {sugg['current_value']:.1f}% to {sugg['suggested_value']:.1f}%"
            elif sugg['feature'] == 'annual_inc':
                rec = f"Increase annual income from ${sugg['current_value']:,.0f} to ${sugg['suggested_value']:,.0f}"
            elif sugg['feature'] == 'delinq_2yrs':
                rec = f"Reduce recent delinquencies (currently {sugg['current_value']:.0f})"
            else:
                rec = f"{sugg['change_direction'].title()} {sugg['feature']} from {sugg['current_value']:.2f} to {sugg['suggested_value']:.2f}"
            
            recommendations.append(rec)
        
        result = {
            'current_probability': current_pred,
            'target_probability': target_pred,
            'suggestions': suggestions,
            'recommendations': recommendations
        }
        
        return result
    
    def create_model_explanation_report(self, test_data: pd.DataFrame,
                                      sample_size: int = 1000) -> Dict[str, Any]:
        """
        Create comprehensive model explanation report
        
        Args:
            test_data: Test dataset
            sample_size: Sample size for SHAP computation
            
        Returns:
            Dictionary with explanation artifacts
        """
        print("Creating comprehensive model explanation report...")
        
        # Sample data for efficiency
        if len(test_data) > sample_size:
            sample_data = test_data.sample(sample_size, random_state=42)
        else:
            sample_data = test_data
        
        # Compute SHAP values
        self.compute_shap_values(sample_data)
        
        # Feature importance
        feature_importance = self.plot_feature_importance()
        
        # SHAP summary plots
        self.plot_shap_summary(sample_data, plot_type='bar')
        self.plot_shap_summary(sample_data, plot_type='dot')
        
        # Individual explanations for high-risk cases
        high_risk_indices = sample_data.index[
            self.model.predict_proba(sample_data)[:, 1] > 0.7
        ].tolist()
        
        if high_risk_indices:
            print(f"Creating individual explanations for {min(3, len(high_risk_indices))} high-risk cases...")
            for i, idx in enumerate(high_risk_indices[:3]):
                sample_idx = sample_data.index.get_loc(idx)
                self.plot_shap_waterfall(sample_idx, sample_data)
        
        # Dependence plots for key features
        key_features = ['dti', 'revol_util', 'annual_inc', 'int_rate']
        available_key_features = [f for f in key_features if f in self.feature_names]
        
        for feature in available_key_features[:3]:  # Limit to avoid too many plots
            self.plot_shap_dependence(feature, sample_data)
        
        report = {
            'feature_importance': feature_importance,
            'shap_values': self.shap_values,
            'sample_data': sample_data,
            'high_risk_cases': high_risk_indices[:10],  # Store top 10
            'key_features_analyzed': available_key_features
        }
        
        return report
