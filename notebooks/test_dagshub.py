import mlflow
import dagshub
import os

# Set your token directly (for testing)
os.environ['DAGSHUB_TOKEN'] = '23519ac2940aab855769c067b98b978ddfe587f3'

try:
    # Initialize DagsHub
    dagshub.init(
        repo_owner='aamir490', 
        repo_name='mlops-mini-project', 
        mlflow=True
    )
    
    print("✅ DagsHub initialized successfully!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # First, let's check if we can create/list experiments
    try:
        # Try to create the experiment first
        mlflow.create_experiment("dvc-pipeline")
        print("✅ Experiment created successfully!")
    except Exception as e:
        print(f"ℹ️ Experiment may already exist: {e}")
    
    # Now try to start a run
    with mlflow.start_run(run_name="test_connection"):
        mlflow.log_param('test_param', 'test_value')
        mlflow.log_metric('test_metric', 0.95)
        print("✅ MLflow connected to DagsHub successfully!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("Running in local mode...")
    
    # Fallback to local MLflow
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    with mlflow.start_run():
        mlflow.log_param('test_param', 'test_value')
        mlflow.log_metric('test_metric', 0.95)
        print("✅ Using local MLflow storage")