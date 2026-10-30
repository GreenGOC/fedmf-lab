import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import pickle

@dataclass
class EvaluationMetrics:
    mse: float
    rmse: float
    mae: float
    r2_score: float
    explained_variance: float
    communication_rounds: Optional[int] = None
    total_bytes: Optional[int] = None
    training_time: Optional[float] = None

class ModelEvaluator:    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_metrics(self, predictions: np.ndarray, 
                         true_values: np.ndarray) -> Dict[str, float]:
        mse = np.mean((predictions - true_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - true_values))
        
        ss_res = np.sum((true_values - predictions) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        explained_variance = 1 - np.var(true_values - predictions) / np.var(true_values)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'explained_variance': explained_variance,
            'predictions_mean': np.mean(predictions),
            'predictions_std': np.std(predictions),
            'true_mean': np.mean(true_values),
            'true_std': np.std(true_values)
        }
    
    def evaluate_model(self, model, test_ratings: List[Tuple], 
                      n_samples: int = 1000) -> EvaluationMetrics:
        if len(test_ratings) > n_samples:
            indices = np.random.choice(len(test_ratings), n_samples, replace=False)
            test_subset = [test_ratings[i] for i in indices]
        else:
            test_subset = test_ratings
        
        predictions = []
        true_values = []
        
        for user_idx, item_idx, rating in test_subset:
            try:
                pred = model.predict(user_idx, item_idx)
                predictions.append(pred)
                true_values.append(rating)
            except:
                predictions.append(0.0)
                true_values.append(rating)
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        metrics = self.calculate_metrics(predictions, true_values)
        
        if hasattr(model, 'coordinator'):
            comm_stats = model.coordinator.get_communication_stats()
            metrics.update(comm_stats)
        
        return EvaluationMetrics(**metrics)
    
    def compare_algorithms(self, results: List[Dict], 
                          metric: str = 'mse') -> pd.DataFrame:
        comparison_data = []
        
        for result in results:
            row = {
                'experiment_id': result['experiment_id'],
                'algorithm': result['config'].algorithm,
                'setting': result['config'].setting,
                'n_parties': result['config'].n_parties,
                'embedding_dim': result['config'].embedding_dim,
                'epsilon': getattr(result['config'], 'epsilon', None),
                'training_time': result['training_time']
            }
            
            # Add evaluation metrics
            for key, value in result['eval_results'].items():
                if isinstance(value, (int, float)):
                    row[key] = value
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_training_curves(self, results: Dict, 
                           save_path: Optional[str] = None):
        """Plot training loss curves"""
        has_predictions = 'predictions' in results['eval_results'] and 'true_values' in results['eval_results']
        
        if has_predictions:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
        else:
            # If no predictions, create a simpler plot with just loss history
            fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            axes = [axes]
        
        # Plot 1: Training loss history
        if 'loss_history' in results['train_results'] and results['train_results']['loss_history']:
            loss_history = results['train_results']['loss_history']
            axes[0].plot(loss_history, label='Training Loss', linewidth=2)
            axes[0].set_xlabel('Global Rounds', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title(f'{results["config"].algorithm} - Training Loss', fontsize=14)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=10)
        else:
            axes[0].text(0.5, 0.5, 'No loss history available', 
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('Training Loss')
        
        if has_predictions:
            predictions = results['eval_results']['predictions']
            true_values = results['eval_results']['true_values']
            errors = predictions - true_values
            
            axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_xlabel('Prediction Error', fontsize=12)
            axes[1].set_ylabel('Frequency', fontsize=12)
            axes[1].set_title('Error Distribution', fontsize=14)
            axes[1].grid(True, alpha=0.3)
            
            axes[2].scatter(true_values, predictions, alpha=0.3, s=10)
            if len(true_values) > 0:
                axes[2].plot([true_values.min(), true_values.max()], 
                            [true_values.min(), true_values.max()], 
                            'r--', alpha=0.5, label='Perfect Prediction')
            axes[2].set_xlabel('True Ratings', fontsize=12)
            axes[2].set_ylabel('Predicted Ratings', fontsize=12)
            axes[2].set_title('True vs Predicted Ratings', fontsize=14)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
            
            metrics_to_plot = ['mse', 'rmse', 'mae', 'r2_score']
            metric_values = [results['eval_results'].get(m, 0) for m in metrics_to_plot]
            
            axes[3].bar(metrics_to_plot, metric_values, alpha=0.7)
            axes[3].set_ylabel('Value', fontsize=12)
            axes[3].set_title('Evaluation Metrics', fontsize=14)
            axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Experiment: {results["experiment_id"]}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close to avoid displaying in non-interactive environments
        else:
            plt.show()
    
    def plot_comparison_summary(self, comparison_df: pd.DataFrame, 
                              metric: str = 'mse',
                              save_path: Optional[str] = None):
        """Plot comparison summary of multiple experiments"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot 1: MSE by algorithm
        sns.boxplot(data=comparison_df, x='algorithm', y=metric, ax=axes[0])
        axes[0].set_title(f'{metric.upper()} by Algorithm')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Training time by algorithm
        if 'training_time' in comparison_df.columns:
            sns.boxplot(data=comparison_df, x='algorithm', y='training_time', ax=axes[1])
            axes[1].set_title('Training Time by Algorithm')
            axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
        
        # Plot 3: MSE vs Number of Parties
        if 'n_parties' in comparison_df.columns:
            axes[2].scatter(comparison_df['n_parties'], comparison_df[metric], alpha=0.6)
            axes[2].set_xlabel('Number of Parties')
            axes[2].set_ylabel(metric.upper())
            axes[2].set_title(f'{metric.upper()} vs Number of Parties')
            axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Privacy-Utility Tradeoff
        privacy_algorithms = comparison_df[comparison_df['algorithm'].str.contains('SGDMF')]
        if not privacy_algorithms.empty and 'epsilon' in privacy_algorithms.columns:
            for algo in privacy_algorithms['algorithm'].unique():
                algo_data = privacy_algorithms[privacy_algorithms['algorithm'] == algo]
                axes[3].plot(algo_data['epsilon'], algo_data[metric], 'o-', label=algo)
            axes[3].set_xlabel('Epsilon (Privacy Budget)')
            axes[3].set_ylabel(metric.upper())
            axes[3].set_title('Privacy-Utility Tradeoff')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
            axes[3].set_xscale('log')
        
        # Plot 5: Communication cost
        if 'total_bytes' in comparison_df.columns:
            comm_data = comparison_df[['algorithm', 'total_bytes']].dropna()
            if not comm_data.empty:
                axes[4].bar(comm_data['algorithm'], comm_data['total_bytes'] / 1e6, alpha=0.7)
                axes[4].set_ylabel('Total Communication (MB)')
                axes[4].set_title('Communication Cost')
                axes[4].set_xticklabels(axes[4].get_xticklabels(), rotation=45)
                axes[4].grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Embedding dimension effect
        if 'embedding_dim' in comparison_df.columns:
            dim_effect = comparison_df.groupby(['algorithm', 'embedding_dim'])[metric].mean().unstack()
            dim_effect.plot(ax=axes[5], marker='o')
            axes[5].set_xlabel('Embedding Dimension')
            axes[5].set_ylabel(metric.upper())
            axes[5].set_title('Effect of Embedding Dimension')
            axes[5].legend(title='Algorithm')
            axes[5].grid(True, alpha=0.3)
        
        plt.suptitle('Federated MF Algorithm Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, results: List[Dict], 
                       output_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("FEDERATED MATRIX FACTORIZATION EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 40)
        
        for result in results:
            report.append(f"\nExperiment: {result['experiment_id']}")
            report.append(f"  Algorithm: {result['config'].algorithm}")
            report.append(f"  Setting: {result['config'].setting}")
            report.append(f"  Dataset: {result['config'].dataset}")
            report.append(f"  Parties: {result['config'].n_parties}")
            
            if hasattr(result['config'], 'epsilon'):
                report.append(f"  Epsilon: {result['config'].epsilon}")
            
            report.append(f"  Training Time: {result['training_time']:.2f}s")
            
            if 'eval_results' in result:
                eval_res = result['eval_results']
                report.append(f"  Test MSE: {eval_res.get('mse', 'N/A'):.4f}")
                report.append(f"  Test RMSE: {eval_res.get('rmse', 'N/A'):.4f}")
                report.append(f"  Test MAE: {eval_res.get('mae', 'N/A'):.4f}")
                report.append(f"  R² Score: {eval_res.get('r2_score', 'N/A'):.4f}")
        
        comparison_df = self.compare_algorithms(results)
        
        report.append("\n\nALGORITHM COMPARISON")
        report.append("-" * 40)
        report.append(comparison_df.to_string())
        
        report.append("\n\nBEST ALGORITHMS")
        report.append("-" * 40)
        
        metrics = ['mse', 'rmse', 'mae', 'r2_score', 'training_time']
        for metric in metrics:
            if metric in comparison_df.columns:
                if metric == 'training_time':
                    best_idx = comparison_df[metric].idxmin()
                else:
                    best_idx = comparison_df[metric].idxmin() if metric != 'r2_score' else comparison_df[metric].idxmax()
                
                best_algo = comparison_df.loc[best_idx, 'algorithm']
                best_value = comparison_df.loc[best_idx, metric]
                report.append(f"Best {metric.upper()}: {best_algo} ({best_value:.4f})")
        
        report_str = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_str)
            
            # Also save comparison as CSV
            comparison_csv = output_path.replace('.txt', '_comparison.csv')
            comparison_df.to_csv(comparison_csv, index=False)
        
        return report_str