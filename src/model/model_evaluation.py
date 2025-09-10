# updated model evaluation

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN")  # Changed from DAGSHUB_PAT to DAGSHUB_TOKEN
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_TOKEN environment variable is not set")

# Set up MLflow tracking URI - use the proper format for DagsHub
repo_owner = "aamir490"  # Replace with your username
repo_name = "mlops-mini-project"  # Replace with your repo name

# Correct MLflow tracking URI format for DagsHub
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

# Set credentials for MLflow
os.environ['MLFLOW_TRACKING_USERNAME'] = repo_owner
os.environ['MLFLOW_TRACKING_PASSWORD'] = dagshub_token

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    # Set experiment name
    mlflow.set_experiment("dvc-pipeline")
    
    # Start MLflow run with simpler configuration
    with mlflow.start_run() as run:
        try:
            # Load model and data
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)
            
            # Save metrics locally
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow (basic metrics only)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters (simplified)
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                # Only log basic parameters to avoid issues
                basic_params = {k: v for k, v in params.items() if not callable(v)}
                for param_name, param_value in basic_params.items():
                    try:
                        mlflow.log_param(param_name, str(param_value))
                    except:
                        pass  # Skip parameters that can't be serialized
            
            # Log model (simplified approach)
            try:
                mlflow.sklearn.log_model(clf, "model")
            except Exception as e:
                logger.warning(f"Model logging failed: {e}")
                # Continue without model logging
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log artifacts (optional - can be skipped if causing issues)
            try:
                mlflow.log_artifact('reports/metrics.json')
                mlflow.log_artifact('reports/experiment_info.json')
            except Exception as e:
                logger.warning(f"Artifact logging failed: {e}")
            
            print("Model evaluation completed successfully!")
            
        except Exception as e:
            logger.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")
            # Ensure the experiment_info.json file is created even if MLflow fails
            try:
                with open('reports/experiment_info.json', 'w') as f:
                    json.dump({'run_id': 'failed', 'model_path': 'failed'}, f)
            except:
                pass

if __name__ == '__main__':
    main()
# updated model evaluation