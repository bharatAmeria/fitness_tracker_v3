import mlflow.pyfunc
import mlflow
import sys
from src.logger import logging
from src.exception import MyException

# Define MLflow experiment & model name
EXPERIMENT_NAME = "exercise_classification"
MODEL_NAME = "Best_Exercise_Classifier"

def get_best_model():
    """
    Fetch the best model from MLflow based on the highest validation metric.
    """
    try:
        logging.info(f"Fetching the best model for {MODEL_NAME} from MLflow...")

        # Connect to MLflow Tracking Server
        client = mlflow.tracking.MlflowClient()

        # Get all registered versions of the model
        model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")

        # Sort models by highest metric (change 'val_accuracy' to your metric)
        best_model = max(
            model_versions,
            key=lambda mv: float(mv.run_id) if mv.run_id else 0  # Ensure valid metric
        )

        best_model_version = best_model.version
        logging.info(f"Best model found: Version {best_model_version}")

        return f"models:/{MODEL_NAME}/{best_model_version}"

    except Exception as e:
        logging.error("ERROR: Failed to fetch the best model from MLflow.")
        raise MyException(e, sys)


def load_model():
    """
    Load the best model from MLflow.
    """
    try:
        best_model_path = get_best_model()
        logging.info(f"Loading model from {best_model_path}...")

        model = mlflow.pyfunc.load_model(best_model_path)
        logging.info("Model loaded successfully!")
        return model

    except Exception as e:
        logging.error("ERROR: Failed to load model from MLflow.")
        raise MyException(e, sys)
