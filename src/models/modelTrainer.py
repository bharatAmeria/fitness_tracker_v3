import sys
import pandas as pd
import mlflow
import time
import mlflow.sklearn

from src.utils.learningAlgorithms import ClassificationAlgorithms
from sklearn.metrics import accuracy_score
from src.constants import *
from src.logger import logging
from src.exception import MyException
from sklearn.model_selection import train_test_split
from src.entity.configEntity import ModelTrainerConfig
from mlflow.models.signature import infer_signature

class ModelTrainer:
    """
    ModelTrainer class for handling model training pipeline.
    
    This class is responsible for:
    1. Loading data
    2. Splitting it into training and testing sets
    3. Preparing feature sets for model training
    """

    def __init__(self, model_trainer_config: ModelTrainerConfig):
        """ 
        Initializes the ModelTrainer class with given configuration.

        Args:
            model_trainer_config (ModelTrainerConfig): Configuration object for model training.
        """
        try:
            logging.info(f"\n{'='*40}\n   Model Training Process Initiated   \n{'='*40}\n")
            logging.info("Initializing the ModelTrainer class and loading configurations...")
            
            self.model_trainer_config = model_trainer_config
            
            logging.info("Configuration loaded successfully. Ready to train the model.\n")
        except Exception as e:
            logging.error("ERROR: Exception occurred during initialization of ModelTrainer class.")
            raise MyException(e, sys)

    def load_data(self):
        """
        Loads data from the specified file.

        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            logging.info("Loading data from the specified file...")
            file_name = self.model_trainer_config.data
            df = pd.read_pickle(file_name)  # Assuming pickle file format
            df_train = df.drop(["participant", "category", "set"], axis=1)
            logging.info(f"Data loaded successfully. Shape: {df.shape}\n")
            return df, df_train
        except Exception as e:
            logging.error("ERROR: Failed to load data.")
            raise MyException(e, sys)

    def split_data_as_train_test(self):
        """
        Splits the dataframe into train and test sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            logging.info("Splitting data into train and test sets...")
            _, df_train = self.load_data()

            # Selecting necessary columns for training
            
            X = df_train.drop(["label"], axis=1)
            y = df_train["label"]

            # Splitting the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            logging.info("Train-test split completed successfully.\n")
      
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("ERROR: Failed to split data into train-test sets.")
            raise MyException(e, sys)

    def prepare_feature_set(self):
        """
        Prepares different feature sets for model training.

        Returns:
            dict: Dictionary containing different feature sets.
        """
        try:
            logging.info("Preparing feature sets...")

            _, df_train = self.load_data()

            basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
            square_features = ["acc_r", "gyr_r"]
            pca_features = ["pca_1", "pca_2", "pca_3"]
            time_features = [f for f in df_train.columns if "_temp_" in f]
            freq_features = [f for f in df_train.columns if ("_freq" in f) or ("_pse" in f)]
            cluster_features = ["cluster"]

            logging.info(f"Basic Features: {len(basic_features)}")
            logging.info(f"Square Features: {len(square_features)}")
            logging.info(f"PCA Features: {len(pca_features)}")
            logging.info(f"Time Features: {len(time_features)}")
            logging.info(f"Frequency Features: {len(freq_features)}")
            logging.info(f"Cluster Features: {len(cluster_features)}")

            feature_set_1 = list(set(basic_features))
            feature_set_2 = list(set(basic_features + square_features + pca_features))
            feature_set_3 = list(set(feature_set_2 + time_features))
            feature_set_4 = list(set(feature_set_3 + freq_features + cluster_features))

            feature_sets = [feature_set_1, feature_set_2, feature_set_3, feature_set_4]

            logging.info("Feature set preparation completed successfully.\n")
            return feature_set_1, feature_set_2, feature_set_3, feature_set_4, feature_sets
        except Exception as e:
            logging.error("ERROR: Failed to prepare feature sets.")
            raise MyException(e, sys)
        
    def grid_search_model_selection(self):
        try:
            learner = ClassificationAlgorithms()
            
            # ✅ Set MLflow Experiment
            mlflow.set_experiment("Model_Selection_Experiment")

            # ✅ Ensure MLflow Tracking URI is Set (Fallback to localhost)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

            X_train, X_test, y_train, y_test = self.split_data_as_train_test()
            selected_features, ordered_features, ordered_scores = learner.forward_selection(MAX_FEATURES, X_train, y_train)
            feature_set_5 = selected_features
            feature_set_1, feature_set_2, feature_set_3, feature_set_4, _ = self.prepare_feature_set()

            possible_feature_sets = [
                list(feature_set_1),
                list(feature_set_2),
                list(feature_set_3),
                list(feature_set_4),
                list(feature_set_5),
            ]

            feature_names = ["Feature Set 1", "Feature Set 2", 
                            "Feature Set 3", "Feature Set 4", 
                            "Selected Features",
                            ]

            best_model = None
            best_model_name = None
            best_accuracy = 0

            for i, f in zip(range(len(possible_feature_sets)), feature_names):
                logging.info(f"Feature set: {i}")
                selected_train_X = X_train[possible_feature_sets[i]]
                selected_test_X = X_test[possible_feature_sets[i]]

                performance_scores = {}

                with mlflow.start_run():  # ✅ Start MLflow run
                    for model_name, model_func in {
                        "NN": learner.feedforward_neural_network,
                        "RF": learner.random_forest,
                        "KNN": learner.k_nearest_neighbor,
                        "DT": learner.decision_tree,
                        "NB": learner.naive_bayes
                    }.items():
                        start_time = time.time()  # ✅ Track Training Time
                        accuracy = 0

                        for it in range(ITERATIONS):
                            logging.info(f"Training {model_name}, iteration: {it}")
                            _, class_test_y, _, _ = model_func(selected_train_X, y_train, selected_test_X, gridsearch=True)
                            accuracy += accuracy_score(y_test, class_test_y)

                        accuracy /= ITERATIONS
                        end_time = time.time()
                        training_duration = end_time - start_time  # ✅ Compute Training Time

                        performance_scores[model_name] = accuracy

                        # ✅ Log Model Metadata to MLflow
                        mlflow.log_param("Feature Set", f)
                        mlflow.log_metric(f"{model_name}_accuracy", accuracy)
                        mlflow.log_metric(f"{model_name}_training_time", training_duration)
                        mlflow.log_param("Number of Features", len(selected_train_X.columns))

                        # ✅ Infer Signature & Log Model
                        signature = infer_signature(selected_train_X, class_test_y)
                        mlflow.sklearn.log_model(learner, f"{model_name}_model", signature=signature)

                        # ✅ Track Best Model
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = learner
                            best_model_name = model_name

                    logging.info(f"🏆 Best Model for Feature Set '{f}': {best_model_name} with Accuracy: {best_accuracy:.4f}")

            # ✅ Register Best Model in MLflow & Move to Staging
            if best_model:
                model_uri = f"models:/{best_model_name}"
                mlflow.register_model(model_uri, best_model_name)

                client = mlflow.tracking.MlflowClient()
                
                # ✅ Automatically Retrieve Latest Model Version
                latest_version = client.get_latest_versions(name=best_model_name, stages=["None"])[0].version

                client.transition_model_version_stage(
                    name=best_model_name,
                    version=latest_version,
                    stage="Staging"
                )

                logging.info(f"✅ Best Model '{best_model_name}' (Version {latest_version}) moved to Staging in MLflow.")

                # ✅ Optional: Auto-Deploy Best Model After Staging
                os.system(f"python deploy_model.py --model_name {best_model_name}")

        except Exception as e:
            logging.error("ERROR: Failed to prepare feature sets.", exc_info=True)
            raise MyException(e, sys)