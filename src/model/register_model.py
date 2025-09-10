



# register_model.py - fixed version

import json
import mlflow
import logging
import os
import dagshub
from datetime import datetime



# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_TOKEN", "23519ac2940aab855769c067b98b978ddfe587f3")

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def setup_mlflow():
    """Setup MLflow with DagsHub or fallback to local"""
    try:
        # Set environment variables
        os.environ["MLFLOW_TRACKING_USERNAME"] = "aamir490"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        # Initialize DagsHub
        dagshub.init(repo_owner="aamir490", repo_name="mlops-mini-project", mlflow=True)
        
        logger.debug("DagsHub MLflow setup successful")
        return True
        
    except Exception as e:
        logger.warning(f"DagsHub setup failed: {e}. Using local MLflow.")
        mlflow.set_tracking_uri("file:///tmp/mlruns")
        return False

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        return {'run_id': 'local_run', 'model_path': './models/model.pkl'}
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        return {'run_id': 'error', 'model_path': './models/model.pkl'}

def register_model_dagshub(model_name: str, model_info: dict):
    """Alternative model registration for DagsHub (without model registry)"""
    try:
        # Since DagsHub doesn't support model registry, we'll use a different approach
        # We can tag the run with model information instead
        
        # Get the run
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(model_info['run_id'])
        
        # Update the run with model tags
        client.set_tag(run.info.run_id, "model_name", model_name)
        client.set_tag(run.info.run_id, "model_stage", "Staging")
        client.set_tag(run.info.run_id, "model_version", "1.0")
        client.set_tag(run.info.run_id, "registered", "true")
        
        logger.debug(f'Model {model_name} tagged in run {run.info.run_id}')
        
        # Also save this information to a local file for tracking
        registry_info = {
            'model_name': model_name,
            'run_id': run.info.run_id,
            'stage': 'Staging',
            'version': '1.0',
            'registered_at': datetime.now().isoformat(),  # FIXED: Use datetime instead of mlflow.utils.time
            'model_uri': f"runs:/{run.info.run_id}/model"
        }
        
        with open('reports/model_registry.json', 'w') as f:
            json.dump(registry_info, f, indent=4)
            
        return registry_info
        
    except Exception as e:
        logger.error('Error during DagsHub model registration: %s', e)
        # Fallback: just save the info locally
        registry_info = {
            'model_name': model_name,
            'run_id': model_info.get('run_id', 'unknown'),
            'stage': 'Staging',
            'version': '1.0',
            'registered_at': datetime.now().isoformat(),
            'error': str(e)
        }
        
        with open('reports/model_registry.json', 'w') as f:
            json.dump(registry_info, f, indent=4)
            
        return registry_info

def register_model_local(model_name: str, model_info: dict):
    """Local model registration without DagsHub"""
    try:
        registry_info = {
            'model_name': model_name,
            'run_id': model_info.get('run_id', 'local_run'),
            'stage': 'Staging',
            'version': '1.0',
            'registered_at': datetime.now().isoformat(),
            'model_path': model_info.get('model_path', './models/model.pkl')
        }
        
        with open('reports/model_registry.json', 'w') as f:
            json.dump(registry_info, f, indent=4)
            
        logger.debug(f'Model {model_name} registered locally')
        return registry_info
        
    except Exception as e:
        logger.error('Error during local model registration: %s', e)
        raise

def main():
    try:
        # Setup MLflow
        use_dagshub = setup_mlflow()
        
        # Load model info
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        
        if use_dagshub:
            # Try DagsHub registration
            try:
                result = register_model_dagshub(model_name, model_info)
                print(f"✅ Model registered with DagsHub: {result}")
            except Exception as e:
                logger.warning(f"DagsHub registration failed: {e}. Falling back to local.")
                result = register_model_local(model_name, model_info)
                print(f"✅ Model registered locally: {result}")
        else:
            # Use local registration
            result = register_model_local(model_name, model_info)
            print(f"✅ Model registered locally: {result}")
            
    except Exception as e:
        logger.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")
        
        # Ensure the registry file is created even if registration fails
        try:
            with open('reports/model_registry.json', 'w') as f:
                json.dump({
                    'model_name': 'my_model',
                    'error': str(e),
                    'registered_at': datetime.now().isoformat()
                }, f)
        except:
            pass

if __name__ == '__main__':
    main()




