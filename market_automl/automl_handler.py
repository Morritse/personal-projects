import os
from google.cloud import automl_v1beta1 as automl
from google.cloud import storage
import pandas as pd
from config import (
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_CLOUD_REGION,
    MODEL_DISPLAY_NAME,
    DATASET_DISPLAY_NAME,
    TECH_STOCKS
)

class AutoMLHandler:
    def __init__(self):
        self.client = automl.TablesClient(
            project=GOOGLE_CLOUD_PROJECT,
            region=GOOGLE_CLOUD_REGION,
        )
        self.storage_client = storage.Client(project=GOOGLE_CLOUD_PROJECT)
        
    def upload_to_gcs(self, local_path, bucket_name, blob_name):
        """Upload a file to Google Cloud Storage"""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{blob_name}"
        
    def create_dataset(self, gcs_path):
        """Create a new dataset in AutoML Tables"""
        dataset = self.client.create_dataset(
            dataset_display_name=DATASET_DISPLAY_NAME,
            gcs_source=gcs_path
        )
        return dataset
        
    def train_model(self, dataset, target_column='target', 
                    train_budget_milli_node_hours=1000):
        """Train an AutoML Tables model"""
        model = self.client.create_model(
            model_display_name=MODEL_DISPLAY_NAME,
            dataset_id=dataset.name,
            train_budget_milli_node_hours=train_budget_milli_node_hours,
            target_column_spec_id=target_column
        )
        return model
        
    def get_model_evaluation(self, model):
        """Get model evaluation metrics"""
        evaluation = self.client.get_model_evaluation(model.name)
        return evaluation
        
    def batch_predict(self, model, gcs_input_path, gcs_output_path):
        """Make batch predictions using the trained model"""
        batch_predict_job = self.client.batch_predict(
            model_name=model.name,
            gcs_source=gcs_input_path,
            gcs_destination_prefix=gcs_output_path
        )
        return batch_predict_job
        
    def process_predictions(self, predictions_path):
        """Process and analyze model predictions"""
        predictions_df = pd.read_csv(predictions_path)
        return predictions_df

def main():
    """Main function to run AutoML workflow"""
    handler = AutoMLHandler()
    
    # Upload data to GCS
    bucket_name = f"{GOOGLE_CLOUD_PROJECT}-automl"
    
    for stock in TECH_STOCKS:
        local_path = f"data/processed/{stock}_features.csv"
        blob_name = f"features/{stock}_features.csv"
        gcs_path = handler.upload_to_gcs(local_path, bucket_name, blob_name)
        
        # Create dataset
        dataset = handler.create_dataset(gcs_path)
        print(f"Created dataset: {dataset.display_name}")
        
        # Train model
        model = handler.train_model(dataset)
        print(f"Training model: {model.display_name}")
        
        # Get evaluation metrics
        evaluation = handler.get_model_evaluation(model)
        print(f"Model evaluation: {evaluation.metrics}")
        
        # Save evaluation metrics
        eval_df = pd.DataFrame({
            'stock': [stock],
            'rmse': [evaluation.regression_evaluation_metrics.root_mean_squared_error],
            'mae': [evaluation.regression_evaluation_metrics.mean_absolute_error],
            'r2': [evaluation.regression_evaluation_metrics.r_squared]
        })
        eval_df.to_csv(f"results/{stock}_model_evaluation.csv", index=False)

if __name__ == "__main__":
    main()
