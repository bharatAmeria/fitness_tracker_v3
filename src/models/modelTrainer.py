import sys
import pandas as pd
import mlflow
import time
import mlflow.sklearn
import joblib
from src.utils.learningAlgorithms import ClassificationAlgorithms
from sklearn.metrics import accuracy_score
from src.constants import *
from src.logger import logging
from src.exception import MyException
from sklearn.model_selection import train_test_split
from src.entity.configEntity import ModelTrainerConfig
from sklearn.ensemble import RandomForestClassifier

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
        
    def model_training(self):
        try:
            learner = ClassificationAlgorithms()
            

            X_train, X_test, y_train, y_test = self.split_data_as_train_test()

            _, _, _, feature_set_4, _ = self.prepare_feature_set()


            start_time = time.time()

            _ ,class_test_y, _, model = learner.random_forest(X_train[feature_set_4], y_train, X_test[feature_set_4], gridsearch=True)
            
            accuracy  = accuracy_score(y_test, class_test_y) * 100

            performance_test_nb = accuracy_score(y_test, class_test_y) * 100

            end_time = time.time()
            training_duration = end_time - start_time

            logging.info(f"Time taken for Training: {training_duration}") 
            logging.info(f"Accuracy achieved: {accuracy} ")
            logging.info(f"Performance on unseen data: {performance_test_nb} ")

            return model

        except Exception as e:
            logging.error("Model selection and training failed.", exc_info=True)
            raise MyException(e, sys)
        
    def save_model(self):
        """
        Saves a trained model using joblib.

        :param model: Trained model to be saved.
        :param model_dir: Directory where the model should be saved.
        :param model_filename: Name of the saved model file (default: "random_forest_model.pkl").
        """
        try:
            logging.info("Saving the model...")
            model = self.model_training()

            os.makedirs(self.model_trainer_config.model, exist_ok=True)

            # Define model path
            model_path = os.path.join(self.model_trainer_config.model, "random_forest_model.pkl")

            # Save the model
            joblib.dump(model, model_path)

            logging.info(f"Model saved successfully at: {model_path}")
        
        except Exception as e:
            logging.error(f"Error saving the model: {e}")
            raise