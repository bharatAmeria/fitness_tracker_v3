import os
import joblib
import sys
import pandas as pd

from src.utils.learningAlgorithms import ClassificationAlgorithms

from sklearn.metrics import accuracy_score, confusion_matrix
from src.constants import *
from src.logger import logging
from src.exception import MyException
from sklearn.model_selection import train_test_split
from src.entity.configEntity import ModelTrainerConfig

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
            score_df = pd.DataFrame()

            
            X_train, X_test, y_train, y_test = self.split_data_as_train_test()
            selected_features, ordered_features, ordered_scores = learner.forward_selection(MAX_FEATURES, X_train, y_train)
            feature_set_5 = selected_features
            feature_set_1, feature_set_2, feature_set_3, feature_set_4, _ = self.prepare_feature_set()

            # Ensure possible feature sets are lists
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
            
            
            for i, f in zip(range(len(possible_feature_sets)), feature_names):
                print("Available columns in X_train:", X_train.columns)
                print("Required columns:", possible_feature_sets[i])
                logging.info(f"Feature set: %s", i)
                selected_train_X = X_train[possible_feature_sets[i]]
                selected_test_X = X_test[possible_feature_sets[i]]

                performance_test_nn = 0
                performance_test_rf = 0

                for it in range(ITERATIONS):
                    logging.info("Training neural network, iteration: %s", it)
                    _, class_test_y, _, _ = learner.feedforward_neural_network(selected_train_X, y_train, selected_test_X, gridsearch=False)
                    performance_test_nn += accuracy_score(y_test, class_test_y)

                    logging.info(f"Training random forest, iteration: %s", it )
                    _, class_test_y, _, _ = learner.random_forest(selected_train_X, y_train, selected_test_X, gridsearch=True)
                    performance_test_rf += accuracy_score(y_test, class_test_y)

                performance_test_nn /= ITERATIONS
                performance_test_rf /= ITERATIONS

                logging.info("\tTraining KNN iteration: %s", it)
                _, class_test_y, _, _ = learner.k_nearest_neighbor(selected_train_X, y_train, selected_test_X, gridsearch=True)
                performance_test_knn = accuracy_score(y_test, class_test_y)

                logging.info("Training decision tree iteration: %s", it)
                _, class_test_y, _, _ = learner.decision_tree(selected_train_X, y_train, selected_test_X, gridsearch=True)
                performance_test_dt = accuracy_score(y_test, class_test_y)

                logging.info("Training naive bayes iteration: %s", it)
                _, class_test_y, _, _ = learner.naive_bayes(selected_train_X, y_train, selected_test_X)
                performance_test_nb = accuracy_score(y_test, class_test_y)

                models = ["NN", "RF", "KNN", "DT", "NB"]
                new_scores = pd.DataFrame({
                    "model": models,
                    "feature_set": f,
                    "accuracy": [
                        performance_test_nn,
                        performance_test_rf,
                        performance_test_knn,
                        performance_test_dt,
                        performance_test_nb,
                    ],
                })
                score_df = pd.concat([score_df, new_scores])

            return score_df
        except Exception as e:
            logging.error("ERROR: Failed to prepare feature sets.")
            raise MyException(e, sys)
        
    
    # def save_model(self, file_path: str) -> None:
    #     """
    #     Save the trained model to a file
        
    #     Args:
    #         file_path (str): Path where to save the model
            
    #     Raises:
    #         MyException: If saving the model fails
    #     """
    #     try:
    #         dir_path = os.path.dirname(file_path)
    #         os.makedirs(dir_path, exist_ok=True)
            
    #         save_object(file_path=file_path, obj=self.model)
    #         logging.info(f"Model saved successfully at: {file_path}")
            
    #     except Exception as e:
    #         logging.error("Error occurred while saving the model")
    #         raise MyException(e, sys)