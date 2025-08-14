"""
Fairness Analysis Module for LendingClub 90-Day Delinquency Prediction
====================================================================

This module provides comprehensive fairness diagnostics and bias detection
for the delinquency prediction model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class FairnessAnalyzer:
    """
    Analyzes model fairness across different demographic groups
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize fairness analyzer
        
        Args:
            model: Trained model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.fairness_metrics = {}
        
    def create_demographic_proxies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic proxy variables for fairness analysis
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with demographic proxies
        """
        df = data.copy()
        
        # Income-based groups
        if 'annual_inc' in df.columns:
            df['income_quartile'] = pd.qcut(
                df['annual_inc'], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
        
        # Geographic proxies (if zip_code available)
        if 'zip_code' in df.columns:
            # Create state-level groupings (simplified)
            high_income_states = ['CA', 'NY', 'CT', 'MA', 'NJ', 'MD', 'WA']
            df['state_income_level'] = df.get('addr_state', 'Unknown').apply(
                lambda x: 'High' if x in high_income_states else 'Other'
            )
        
        # Loan purpose groups
        if 'purpose' in df.columns:
            essential_purposes = ['medical', 'car', 'home_improvement', 'major_purchase']
            df['loan_purpose_type'] = df['purpose'].apply(
                lambda x: 'Essential' if x in essential_purposes else 'Discretionary'
            )
        
        # Credit age groups (proxy for age)
        if 'credit_age_months' in df.columns:
            df['credit_maturity'] = pd.cut(
                df['credit_age_months'], 
                bins=[0, 60, 120, 240, float('inf')],
                labels=['New', 'Developing', 'Established', 'Mature']
            )
        
        return df
    
    def compute_group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray, groups: pd.Series,
                            group_name: str) -> Dict[str, Any]:
        """
        Compute fairness metrics for different groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            groups: Group assignments
            group_name: Name of the grouping variable
            
        Returns:
            Dictionary of fairness metrics by group
        """
        results = {
            'group_name': group_name,
            'group_metrics': {},
            'parity_gaps': {}
        }
        
        unique_groups = groups.dropna().unique()
        
        for group in unique_groups:
            mask = groups == group
            if mask.sum() == 0:
                continue
                
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            group_y_proba = y_pred_proba[mask]
            
            # Basic metrics
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            
            metrics = {
                'sample_size': len(group_y_true),
                'base_rate': group_y_true.mean(),
                'positive_rate': group_y_pred.mean(),
                'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # Sensitivity/Recall
                'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # Specificity
                'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
                'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Precision
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
                'mean_prediction': group_y_proba.mean()
            }
            
            # AUC if we have enough samples and variation
            if len(np.unique(group_y_true)) > 1 and len(group_y_true) > 10:
                try:
                    fpr_curve, tpr_curve, _ = roc_curve(group_y_true, group_y_proba)
                    metrics['auc'] = auc(fpr_curve, tpr_curve)
                except:
                    metrics['auc'] = np.nan
            else:
                metrics['auc'] = np.nan
            
            results['group_metrics'][group] = metrics
        
        # Compute parity gaps
        if len(results['group_metrics']) >= 2:
            groups_list = list(results['group_metrics'].keys())
            reference_group = groups_list[0]  # Use first group as reference
            
            for metric in ['tpr', 'tnr', 'fpr', 'ppv', 'positive_rate', 'mean_prediction']:
                ref_value = results['group_metrics'][reference_group][metric]
                gaps = {}
                
                for group in groups_list[1:]:
                    group_value = results['group_metrics'][group][metric]
                    gaps[f"{reference_group}_vs_{group}"] = abs(ref_value - group_value)
                
                results['parity_gaps'][metric] = gaps
        
        return results
    
    def analyze_fairness(self, data: pd.DataFrame, y_true: np.ndarray,
                        threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive fairness analysis
        
        Args:
            data: Input data with demographic proxies
            y_true: True labels
            threshold: Classification threshold
            
        Returns:
            Dictionary with fairness analysis results
        """
        print("Performing comprehensive fairness analysis...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(data[self.feature_names])[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Create demographic proxies
        data_with_proxies = self.create_demographic_proxies(data)
        
        fairness_results = {}
        
        # Analyze different groupings
        grouping_variables = [
            'income_quartile', 'state_income_level', 
            'loan_purpose_type', 'credit_maturity'
        ]
        
        for group_var in grouping_variables:
            if group_var in data_with_proxies.columns:
                print(f"Analyzing fairness for {group_var}...")
                
                group_results = self.compute_group_metrics(
                    y_true, y_pred, y_pred_proba,
                    data_with_proxies[group_var], group_var
                )
                
                fairness_results[group_var] = group_results
        
        # Overall fairness summary
        fairness_summary = self.create_fairness_summary(fairness_results)
        
        return {
            'detailed_results': fairness_results,
            'summary': fairness_summary,
            'threshold_used': threshold
        }
    
    def create_fairness_summary(self, fairness_results: Dict) -> Dict[str, Any]:
        """
        Create summary of fairness analysis
        
        Args:
            fairness_results: Detailed fairness results
            
        Returns:
            Summary dictionary
        """
        summary = {
            'max_tpr_gap': 0,
            'max_fpr_gap': 0,
            'max_ppv_gap': 0,
            'problematic_groups': [],
            'recommendations': []
        }
        
        # Find maximum gaps across all groupings
        for group_name, results in fairness_results.items():
            if 'parity_gaps' in results:
                for metric, gaps in results['parity_gaps'].items():
                    max_gap = max(gaps.values()) if gaps else 0
                    
                    if metric == 'tpr' and max_gap > summary['max_tpr_gap']:
                        summary['max_tpr_gap'] = max_gap
                    elif metric == 'fpr' and max_gap > summary['max_fpr_gap']:
                        summary['max_fpr_gap'] = max_gap
                    elif metric == 'ppv' and max_gap > summary['max_ppv_gap']:
                        summary['max_ppv_gap'] = max_gap
        
        # Identify problematic groups (gaps > 0.1)
        threshold_gap = 0.1
        
        for group_name, results in fairness_results.items():
            if 'parity_gaps' in results:
                for metric, gaps in results['parity_gaps'].items():
                    for comparison, gap in gaps.items():
                        if gap > threshold_gap:
                            summary['problematic_groups'].append({
                                'grouping': group_name,
                                'comparison': comparison,
                                'metric': metric,
                                'gap': gap
                            })
        
        # Generate recommendations
        if summary['max_tpr_gap'] > 0.1:
            summary['recommendations'].append(
                "Consider threshold adjustment to reduce True Positive Rate disparities"
            )
        
        if summary['max_fpr_gap'] > 0.1:
            summary['recommendations'].append(
                "Monitor False Positive Rate differences across groups"
            )
        
        if len(summary['problematic_groups']) > 0:
            summary['recommendations'].append(
                "Implement bias mitigation techniques or post-processing adjustments"
            )
        else:
            summary['recommendations'].append(
                "Model shows acceptable fairness metrics across analyzed groups"
            )
        
        return summary
    
    def plot_fairness_metrics(self, fairness_results: Dict[str, Any]):
        """
        Plot fairness metrics visualization
        
        Args:
            fairness_results: Results from fairness analysis
        """
        print("Creating fairness visualization...")
        
        # Prepare data for plotting
        plot_data = []
        
        for group_name, results in fairness_results['detailed_results'].items():
            for group, metrics in results['group_metrics'].items():
                plot_data.append({
                    'grouping': group_name,
                    'group': group,
                    'tpr': metrics['tpr'],
                    'fpr': metrics['fpr'],
                    'ppv': metrics['ppv'],
                    'sample_size': metrics['sample_size'],
                    'base_rate': metrics['base_rate']
                })
        
        if not plot_data:
            print("No data available for plotting")
            return
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fairness Metrics Across Groups', fontsize=16)
        
        # TPR by group
        ax1 = axes[0, 0]
        for grouping in df_plot['grouping'].unique():
            group_data = df_plot[df_plot['grouping'] == grouping]
            ax1.scatter(group_data['group'], group_data['tpr'], 
                       label=grouping, s=group_data['sample_size']/10, alpha=0.7)
        ax1.set_title('True Positive Rate by Group')
        ax1.set_ylabel('TPR')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # FPR by group
        ax2 = axes[0, 1]
        for grouping in df_plot['grouping'].unique():
            group_data = df_plot[df_plot['grouping'] == grouping]
            ax2.scatter(group_data['group'], group_data['fpr'], 
                       label=grouping, s=group_data['sample_size']/10, alpha=0.7)
        ax2.set_title('False Positive Rate by Group')
        ax2.set_ylabel('FPR')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision by group
        ax3 = axes[1, 0]
        for grouping in df_plot['grouping'].unique():
            group_data = df_plot[df_plot['grouping'] == grouping]
            ax3.scatter(group_data['group'], group_data['ppv'], 
                       label=grouping, s=group_data['sample_size']/10, alpha=0.7)
        ax3.set_title('Precision by Group')
        ax3.set_ylabel('Precision')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Base rate by group
        ax4 = axes[1, 1]
        for grouping in df_plot['grouping'].unique():
            group_data = df_plot[df_plot['grouping'] == grouping]
            ax4.scatter(group_data['group'], group_data['base_rate'], 
                       label=grouping, s=group_data['sample_size']/10, alpha=0.7)
        ax4.set_title('Base Rate (Actual Positive Rate) by Group')
        ax4.set_ylabel('Base Rate')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        print("\nFairness Summary:")
        print("=" * 50)
        
        summary = fairness_results['summary']
        print(f"Maximum TPR Gap: {summary['max_tpr_gap']:.3f}")
        print(f"Maximum FPR Gap: {summary['max_fpr_gap']:.3f}")
        print(f"Maximum Precision Gap: {summary['max_ppv_gap']:.3f}")
        
        if summary['problematic_groups']:
            print(f"\nProblematic Groups ({len(summary['problematic_groups'])}):")
            for issue in summary['problematic_groups']:
                print(f"  - {issue['grouping']}: {issue['comparison']} "
                      f"({issue['metric']} gap: {issue['gap']:.3f})")
        
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  - {rec}")
    
    def threshold_analysis(self, data: pd.DataFrame, y_true: np.ndarray,
                          group_column: str, thresholds: List[float] = None) -> Dict:
        """
        Analyze fairness metrics across different decision thresholds
        
        Args:
            data: Input data
            y_true: True labels
            group_column: Column name for grouping
            thresholds: List of thresholds to analyze
            
        Returns:
            Dictionary with threshold analysis results
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 0.9, 0.1)
        
        print(f"Analyzing fairness across thresholds for {group_column}...")
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(data[self.feature_names])[:, 1]
        
        # Create demographic proxies
        data_with_proxies = self.create_demographic_proxies(data)
        
        if group_column not in data_with_proxies.columns:
            print(f"Group column {group_column} not found")
            return {}
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            group_metrics = self.compute_group_metrics(
                y_true, y_pred, y_pred_proba,
                data_with_proxies[group_column], group_column
            )
            
            # Extract key metrics for each group
            for group, metrics in group_metrics['group_metrics'].items():
                results.append({
                    'threshold': threshold,
                    'group': group,
                    'tpr': metrics['tpr'],
                    'fpr': metrics['fpr'],
                    'ppv': metrics['ppv'],
                    'positive_rate': metrics['positive_rate']
                })
        
        # Plot threshold analysis
        df_thresh = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Fairness Metrics vs Threshold - {group_column}', fontsize=16)
        
        metrics_to_plot = ['tpr', 'fpr', 'ppv', 'positive_rate']
        metric_names = ['True Positive Rate', 'False Positive Rate', 'Precision', 'Positive Rate']
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i//2, i%2]
            
            for group in df_thresh['group'].unique():
                group_data = df_thresh[df_thresh['group'] == group]
                ax.plot(group_data['threshold'], group_data[metric], 
                       marker='o', label=group, linewidth=2)
            
            ax.set_title(name)
            ax.set_xlabel('Threshold')
            ax.set_ylabel(name)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'threshold_results': df_thresh,
            'group_column': group_column
        }
