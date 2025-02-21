import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
import logging
from pathlib import Path
from datetime import datetime

from .validation import OverfitDetector
from .main import run_backtest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data() -> Dict[str, pd.DataFrame]:
    """Load data for analysis."""
    # Use fewer symbols for quicker validation
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data_dict = {}
    
    for symbol in symbols:
        filepath = f'momentum_ai_trading/data/processed/{symbol}_processed_daily.csv'
        logger.info(f"Loading data from: {filepath}")
        data = pd.read_csv(filepath)
        # Use last 2 years of data for validation
        data_dict[symbol] = data.tail(500)
        
    return data_dict

def run_overfitting_analysis():
    """Run comprehensive overfitting analysis."""
    logger.info("\nStarting Overfitting Analysis...")
    logger.info("================================")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'momentum_ai_trading/results/validation/run_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_dict = load_data()
    
    # Initialize detector
    detector = OverfitDetector(window_size=60, step_size=1)  # Reduced window size
    
    # 1. Walk-forward Analysis
    logger.info("\nPerforming Walk-forward Analysis...")
    wf_results = detector.walk_forward_analysis(data_dict, n_splits=3)  # Fewer splits
    
    # Plot walk-forward results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(wf_results['returns'], marker='o')
    plt.title('Returns Across Time Periods')
    plt.xlabel('Period')
    plt.ylabel('Return')
    
    plt.subplot(2, 2, 2)
    plt.plot(wf_results['sharpe_ratios'], marker='o', color='green')
    plt.title('Sharpe Ratios Across Time Periods')
    plt.xlabel('Period')
    plt.ylabel('Sharpe Ratio')
    
    plt.subplot(2, 2, 3)
    plt.plot(wf_results['max_drawdowns'], marker='o', color='red')
    plt.title('Max Drawdowns Across Time Periods')
    plt.xlabel('Period')
    plt.ylabel('Max Drawdown')
    
    plt.subplot(2, 2, 4)
    plt.plot(wf_results['prediction_stability'], marker='o', color='purple')
    plt.title('Prediction Stability Across Time Periods')
    plt.xlabel('Period')
    plt.ylabel('Stability Score')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'walk_forward_analysis.png')
    
    # 2. Feature Importance Analysis
    logger.info("\nAnalyzing Feature Importance...")
    feature_importance = detector.feature_importance_analysis(data_dict)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(feature_importance.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Stability')
    plt.savefig(results_dir / 'feature_importance.png')
    
    # 3. Prediction Consistency Analysis
    logger.info("\nAnalyzing Prediction Consistency...")
    consistency_metrics = detector.prediction_consistency_analysis(data_dict)
    
    plt.figure(figsize=(10, 6))
    plt.bar(consistency_metrics.keys(), consistency_metrics.values())
    plt.title('Model Prediction Agreement by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Agreement Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / 'prediction_consistency.png')
    
    # Save numerical results
    results_summary = {
        'walk_forward': pd.DataFrame(wf_results),
        'consistency': pd.Series(consistency_metrics)
    }
    
    with open(results_dir / 'analysis_summary.txt', 'w') as f:
        f.write("Overfitting Analysis Summary\n")
        f.write("===========================\n\n")
        
        f.write("Walk-forward Analysis:\n")
        f.write("-----------------\n")
        f.write(f"Mean Return: {np.mean(wf_results['returns']):.2%}\n")
        f.write(f"Return Std: {np.std(wf_results['returns']):.2%}\n")
        f.write(f"Mean Sharpe: {np.mean(wf_results['sharpe_ratios']):.2f}\n")
        f.write(f"Sharpe Std: {np.std(wf_results['sharpe_ratios']):.2f}\n")
        f.write(f"Mean Max DD: {np.mean(wf_results['max_drawdowns']):.2%}\n")
        f.write(f"Max DD Std: {np.std(wf_results['max_drawdowns']):.2%}\n\n")
        
        f.write("Prediction Consistency:\n")
        f.write("----------------------\n")
        for symbol, agreement in consistency_metrics.items():
            f.write(f"{symbol}: {agreement:.2f}\n")
        
        # Add warning indicators
        f.write("\nPotential Overfitting Indicators:\n")
        f.write("--------------------------------\n")
        
        # Check return consistency
        return_std = np.std(wf_results['returns'])
        if return_std > 0.1:  # High return variability
            f.write("WARNING: High variability in returns across periods\n")
            
        # Check Sharpe ratio consistency
        sharpe_std = np.std(wf_results['sharpe_ratios'])
        if sharpe_std > 0.5:  # High Sharpe ratio variability
            f.write("WARNING: High variability in Sharpe ratios across periods\n")
            
        # Check prediction stability
        mean_stability = np.mean(wf_results['prediction_stability'])
        if mean_stability > 0.5:  # High prediction instability
            f.write("WARNING: Low prediction stability across models\n")
            
        # Check model agreement
        mean_agreement = np.mean(list(consistency_metrics.values()))
        if mean_agreement < 0.6:  # Low model agreement
            f.write("WARNING: Low agreement between different model types\n")
    
    logger.info(f"\nAnalysis complete. Results saved to: {results_dir}")
    
    return results_summary

if __name__ == "__main__":
    run_overfitting_analysis()
