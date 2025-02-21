import os
import pandas as pd
import matplotlib.pyplot as plt
from data_collection import DataCollector
from feature_engineering import FeatureEngineer
from automl_handler import AutoMLHandler
from config import TECH_STOCKS, INDEX_FUNDS

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data/raw', 'data/processed', 'results', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def plot_correlations(features_dict, output_dir='results'):
    """Plot correlation heatmaps for each stock"""
    for stock in features_dict:
        plt.figure(figsize=(12, 8))
        corr_matrix = features_dict[stock].corr()
        plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.title(f'Feature Correlations - {stock}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{stock}_correlations.png')
        plt.close()

def plot_feature_importance(model_evaluations, output_dir='results'):
    """Plot feature importance for each stock's model"""
    for stock in model_evaluations:
        plt.figure(figsize=(10, 6))
        importance_df = pd.DataFrame({
            'feature': model_evaluations[stock]['feature_importance'].keys(),
            'importance': model_evaluations[stock]['feature_importance'].values()
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title(f'Feature Importance - {stock}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{stock}_feature_importance.png')
        plt.close()

def main():
    """Main execution function"""
    print("Setting up directories...")
    setup_directories()
    
    print("Collecting data from Alpaca...")
    collector = DataCollector()
    data = collector.fetch_all_data()
    collector.save_data(data, 'data/raw')
    
    print("Engineering features...")
    engineer = FeatureEngineer(data)
    features = engineer.engineer_features()
    automl_features = engineer.prepare_for_automl()
    
    # Save processed features
    for symbol, df in automl_features.items():
        df.to_csv(f'data/processed/{symbol}_features.csv')
    
    print("Training AutoML models...")
    handler = AutoMLHandler()
    model_evaluations = {}
    
    for stock in TECH_STOCKS:
        print(f"Processing {stock}...")
        # Upload and train
        local_path = f"data/processed/{stock}_features.csv"
        bucket_name = f"{handler.client.project}-automl"
        blob_name = f"features/{stock}_features.csv"
        
        try:
            gcs_path = handler.upload_to_gcs(local_path, bucket_name, blob_name)
            dataset = handler.create_dataset(gcs_path)
            model = handler.train_model(dataset)
            evaluation = handler.get_model_evaluation(model)
            
            # Store evaluation results
            model_evaluations[stock] = {
                'rmse': evaluation.regression_evaluation_metrics.root_mean_squared_error,
                'mae': evaluation.regression_evaluation_metrics.mean_absolute_error,
                'r2': evaluation.regression_evaluation_metrics.r_squared,
                'feature_importance': evaluation.feature_importance
            }
            
        except Exception as e:
            print(f"Error processing {stock}: {str(e)}")
    
    # Save evaluation results
    eval_df = pd.DataFrame(model_evaluations).T
    eval_df.to_csv('results/model_evaluations.csv')
    
    # Create visualizations
    print("Creating visualizations...")
    plot_correlations(automl_features)
    plot_feature_importance(model_evaluations)
    
    print("Analysis complete! Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
