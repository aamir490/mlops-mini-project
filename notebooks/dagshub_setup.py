import mlflow
import dagshub

dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)
  
  print("✅ MLflow connected to Dagshub!")
  
  