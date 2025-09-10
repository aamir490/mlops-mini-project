# src/model/model_evaluation.py
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub

# -------------------------------
# Setup Logging
# -------------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# -------------------------------
# DagsHub / MLflow Setup
# -------------------------------
dagshub_token = os.getenv("DAGSHUB_PAT")

if dagshub_token:
    # If token exists → configure remote MLflow
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)
    mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

    logger.info("✅ Using DagsHub MLflow tracking")
else:
    # Fallback → local MLflow
    local_path = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file:///{local_path}")
    logger.info("⚠️ DAGSHUB_PAT not found. Using local MLflow tracking at %s", local_path)


# -------------------------------
# Utility Functions
# -------------------------------
def load_model(file_path: str):
    """Load the trained model from a file."""
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', file_path)
        return model
    except Exception as e:
        logger.error('Error loading model: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data: %s', e)
        raise


def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.debug('Model evaluation metrics calculated')
        return metrics
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def save_json(data: dict, file_path: str):
    """Save dictionary as JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug('Saved JSON to %s', file_path)
    except Exception as e:
        logger.error('Error saving JSON: %s', e)
        raise


# -------------------------------
# Main
# -------------------------------
def main():
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        try:
            # Load model + test data
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate
            metrics = evaluate_model(clf, X_test, y_test)

            # Save metrics to file
            save_json(metrics, 'reports/metrics.json')

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model parameters
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for k, v in params.items():
                    mlflow.log_param(k, v)

            # ✅ Log & register model in one step
            model_name = "mlops-mini-project-model"
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="sklearn-model",
                registered_model_name=model_name
            )
            # Save model locally too (so DVC can track it if needed)
            os.makedirs("reports", exist_ok=True)
            with open("reports/model.pkl", "wb") as f:
                pickle.dump(clf, f)
            mlflow.log_artifact("reports/model.pkl")

            # Save experiment info
            experiment_info = {'run_id': run.info.run_id, 'model_path': "sklearn-model"}
            save_json(experiment_info, 'reports/experiment_info.json')

            # Log artifacts
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/experiment_info.json')
            mlflow.log_artifact('model_evaluation_errors.log')

            logger.info("✅ Model evaluation completed successfully")

        except Exception as e:
            logger.error("Failed script execution: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()



















# import os
# import json
# import pickle
# import logging
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import mlflow
# import mlflow.sklearn
# import dagshub

# # -------------------------------
# # Setup Logging
# # -------------------------------
# logger = logging.getLogger('model_evaluation')
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel(logging.ERROR)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)


# # -------------------------------
# # DagsHub / MLflow Setup
# # -------------------------------
# dagshub_token = os.getenv("DAGSHUB_PAT")

# if dagshub_token:
#     # If token exists → configure remote MLflow
#     os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
#     os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

#     dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)
#     mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

#     logger.info("✅ Using DagsHub MLflow tracking")
# else:
#     # Fallback → local MLflow
#     local_path = os.path.abspath("mlruns")
#     mlflow.set_tracking_uri(f"file:///{local_path}")
#     logger.info("⚠️ DAGSHUB_PAT not found. Using local MLflow tracking at %s", local_path)


# # -------------------------------
# # Utility Functions
# # -------------------------------
# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', file_path)
#         return model
#     except Exception as e:
#         logger.error('Error loading model: %s', e)
#         raise


# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s', file_path)
#         return df
#     except Exception as e:
#         logger.error('Error loading data: %s', e)
#         raise


# def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
#     """Evaluate the model and return the evaluation metrics."""
#     try:
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]

#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred),
#             'recall': recall_score(y_test, y_pred),
#             'auc': roc_auc_score(y_test, y_pred_proba)
#         }
#         logger.debug('Model evaluation metrics calculated')
#         return metrics
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise


# def save_json(data: dict, file_path: str):
#     """Save dictionary as JSON file."""
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, 'w') as f:
#             json.dump(data, f, indent=4)
#         logger.debug('Saved JSON to %s', file_path)
#     except Exception as e:
#         logger.error('Error saving JSON: %s', e)
#         raise


# # -------------------------------
# # Main
# # -------------------------------
# def main():
#     mlflow.set_experiment("dvc-pipeline")

#     with mlflow.start_run() as run:
#         try:
#             # Load model + test data
#             clf = load_model('./models/model.pkl')
#             test_data = load_data('./data/processed/test_bow.csv')

#             X_test = test_data.iloc[:, :-1].values
#             y_test = test_data.iloc[:, -1].values

#             # Evaluate
#             metrics = evaluate_model(clf, X_test, y_test)

#             # Save metrics to file
#             save_json(metrics, 'reports/metrics.json')

#             # Log metrics to MLflow
#             for k, v in metrics.items():
#                 mlflow.log_metric(k, v)

#             # Log model parameters
#             if hasattr(clf, 'get_params'):
#                 params = clf.get_params()
#                 for k, v in params.items():
#                     mlflow.log_param(k, v)

#             # Log model to MLflow
#             mlflow.sklearn.log_model(clf, "model")

#             # Save experiment info
#             experiment_info = {'run_id': run.info.run_id, 'model_path': "model"}
#             save_json(experiment_info, 'reports/experiment_info.json')

#             # Log artifacts
#             mlflow.log_artifact('reports/metrics.json')
#             mlflow.log_artifact('reports/experiment_info.json')
#             mlflow.log_artifact('model_evaluation_errors.log')

#             logger.info("✅ Model evaluation completed successfully")

#         except Exception as e:
#             logger.error("Failed script execution: %s", e)
#             print(f"Error: {e}")


# if __name__ == "__main__":
#     main()



# # ---------------------------
# # Complete MLflow + DagsHub Script with Token Authentication
# # ---------------------------

# import os
# import pickle
# import json
# import logging
# import re
# import string
# import numpy as np
# import pandas as pd
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# import mlflow
# import mlflow.sklearn
# import dagshub

# # ---------------------------
# # 1️⃣ DagsHub Token Setup
# # ---------------------------
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# # Set MLflow credentials for DagsHub
# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# # Initialize DagsHub repo
# dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)

# # Set MLflow tracking URI
# mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

# # ---------------------------
# # 2️⃣ Logging Configuration
# # ---------------------------
# logger = logging.getLogger('model_evaluation')
# logger.setLevel(logging.DEBUG)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel(logging.ERROR)

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# # ---------------------------
# # 3️⃣ Text Preprocessing Functions
# # ---------------------------
# def lemmatization(text):
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(word) for word in text]
#     return " ".join(text)

# def remove_stop_words(text):
#     stop_words = set(stopwords.words("english"))
#     text = [word for word in str(text).split() if word not in stop_words]
#     return " ".join(text)

# def remove_numbers(text):
#     text = ''.join([char for char in text if not char.isdigit()])
#     return text

# def lower_case(text):
#     text = text.split()
#     text = [word.lower() for word in text]
#     return " ".join(text)

# def remove_punctuations(text):
#     text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
#     text = text.replace('؛', "")
#     text = re.sub('\s+', ' ', text).strip()
#     return text

# def remove_urls(text):
#     url_pattern = re.compile(r'https?://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def normalize_text(df):
#     """Normalize the text data."""
#     df['content'] = df['content'].apply(lower_case)
#     df['content'] = df['content'].apply(remove_stop_words)
#     df['content'] = df['content'].apply(remove_numbers)
#     df['content'] = df['content'].apply(remove_punctuations)
#     df['content'] = df['content'].apply(remove_urls)
#     df['content'] = df['content'].apply(lemmatization)
#     return df

# # ---------------------------
# # 4️⃣ Utility Functions
# # ---------------------------
# def evaluate_model(clf, X_test, y_test) -> dict:
#     """Evaluate model and return metrics"""
#     y_pred = clf.predict(X_test)
#     y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else np.zeros_like(y_pred)
    
#     metrics = {
#         'accuracy': accuracy_score(y_test, y_pred),
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1_score': f1_score(y_test, y_pred),
#         'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba.any() else 0.0
#     }
#     return metrics

# def save_json(data: dict, file_path: str):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     with open(file_path, 'w') as f:
#         json.dump(data, f, indent=4)

# # ---------------------------
# # 5️⃣ Main Script
# # ---------------------------
# def main():
#     try:
#         # Load dataset
#         df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

#         # Keep only 'happiness' and 'sadness'
#         df = df[df['sentiment'].isin(['happiness','sadness'])]
#         df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1})

#         # Normalize text
#         df = normalize_text(df)

#         # Feature extraction
#         vectorizer = CountVectorizer()
#         X = vectorizer.fit_transform(df['content'])
#         y = df['sentiment'].values

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Hyperparameter tuning for Logistic Regression
#         param_grid = {
#             'C': [0.1, 1, 10],
#             'penalty': ['l1', 'l2'],
#             'solver': ['liblinear']
#         }

#         grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
#         grid_search.fit(X_train, y_train)

#         # Start MLflow run
#         with mlflow.start_run(run_name="LoR_Hyperparameter_Tuning") as run:
#             best_model = grid_search.best_estimator_

#             # Evaluate
#             metrics = evaluate_model(best_model, X_test, y_test)

#             # Log metrics
#             for k, v in metrics.items():
#                 mlflow.log_metric(k, v)

#             # Log best model
#             mlflow.sklearn.log_model(best_model, name="best_model")

#             # Save metrics and model info locally
#             save_json(metrics, 'reports/metrics.json')
#             save_json({'run_id': run.info.run_id, 'model_name': 'best_model'}, 'reports/experiment_info.json')

#             # Log artifacts
#             mlflow.log_artifact('reports/metrics.json')
#             mlflow.log_artifact('reports/experiment_info.json')
#             if os.path.exists('model_evaluation_errors.log'):
#                 mlflow.log_artifact('model_evaluation_errors.log')

#             print("✅ Model training, evaluation, and logging completed successfully!")
#             print("Metrics:", metrics)
#             print("Run info logged to DagsHub MLflow")

#     except Exception as e:
#         logger.error(f"Failed script execution: {e}")
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()


# # updated model evaluation


# import numpy as np
# import pandas as pd
# import pickle
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import logging
# import mlflow
# import mlflow.sklearn
# import dagshub
# import os

# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "aamir490"
# repo_name = "mlops-mini-project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# # dagshub.init(repo_owner='aamir490', repo_name='mlops-mini-project', mlflow=True)
# # mlflow.set_tracking_uri('https://dagshub.com/aamir490/mlops-mini-project.mlflow')

# # logging configuration
# logger = logging.getLogger('model_evaluation')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel('ERROR')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', file_path)
#         return model
#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the model: %s', e)
#         raise

# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
#     """Evaluate the model and return the evaluation metrics."""
#     try:
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_pred_proba)

#         metrics_dict = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'auc': auc
#         }
#         logger.debug('Model evaluation metrics calculated')
#         return metrics_dict
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise

# def save_metrics(metrics: dict, file_path: str) -> None:
#     """Save the evaluation metrics to a JSON file."""
#     try:
#         with open(file_path, 'w') as file:
#             json.dump(metrics, file, indent=4)
#         logger.debug('Metrics saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the metrics: %s', e)
#         raise

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         model_info = {'run_id': run_id, 'model_path': model_path}
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise

# def main():
#     mlflow.set_experiment("dvc-pipeline")
#     with mlflow.start_run() as run:  # Start an MLflow run
#         try:
#             clf = load_model('./models/model.pkl')
#             test_data = load_data('./data/processed/test_bow.csv')
            
#             X_test = test_data.iloc[:, :-1].values
#             y_test = test_data.iloc[:, -1].values

#             metrics = evaluate_model(clf, X_test, y_test)
            
#             save_metrics(metrics, 'reports/metrics.json')
            
#             # Log metrics to MLflow
#             for metric_name, metric_value in metrics.items():
#                 mlflow.log_metric(metric_name, metric_value)
            
#             # Log model parameters to MLflow
#             if hasattr(clf, 'get_params'):
#                 params = clf.get_params()
#                 for param_name, param_value in params.items():
#                     mlflow.log_param(param_name, param_value)
            
#             # Log model to MLflow
#             mlflow.sklearn.log_model(clf, "model")
            
#             # Save model info
#             save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
#             # Log the metrics file to MLflow
#             mlflow.log_artifact('reports/metrics.json')

#             # Log the model info file to MLflow
#             mlflow.log_artifact('reports/model_info.json')

#             # Log the evaluation errors log file to MLflow
#             mlflow.log_artifact('model_evaluation_errors.log')
#         except Exception as e:
#             logger.error('Failed to complete the model evaluation process: %s', e)
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()
