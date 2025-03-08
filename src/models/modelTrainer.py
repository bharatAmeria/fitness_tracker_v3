import os
import joblib
import sys
import pandas as pd

from src.utils.learningAlgorithms import ClassificationAlgorithms
import itertools
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
            self.df = self.load_data()
            
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
            self.df = pd.read_pickle(file_name)  # Assuming pickle file format
            
            logging.info(f"Data loaded successfully. Shape: {self.df.shape}\n")
            return self.df
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

            if self.df is None:
                logging.info("Data is not loaded. Loading data first...")
                self.load_data()

            # Selecting necessary columns for training
            df_filtered = self.df[["participant", "category", "set", "label"]].copy()

            # Defining X (features) and y (target)
            X = df_filtered.drop(columns=["label"])
            y = df_filtered["label"]

            # Splitting the dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            logging.info("Train-test split completed successfully.\n")
            return self.X_train, self.X_test, self.y_train, self.y_test
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

            if self.df is None:
                logging.info("Data is not loaded. Loading data first...")
                self.load_data()

            basic_features = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
            square_features = ["acc_r", "gyr_r"]
            pca_features = ["pca_1", "pca_2", "pca_3"]
            time_features = [f for f in self.df.columns if "_temp_" in f]
            freq_features = [f for f in self.df.columns if ("_freq" in f) or ("_pse" in f)]
            cluster_features = ["cluster"]

            logging.info(f"Basic Features: {len(basic_features)}")
            logging.info(f"Square Features: {len(square_features)}")
            logging.info(f"PCA Features: {len(pca_features)}")
            logging.info(f"Time Features: {len(time_features)}")
            logging.info(f"Frequency Features: {len(freq_features)}")
            logging.info(f"Cluster Features: {len(cluster_features)}")

            self.feature_set_1 = list(set(basic_features))
            self.feature_set_2 = list(set(basic_features + square_features + pca_features))
            self.feature_set_3 = list(set(self.feature_set_2 + time_features))
            self.feature_set_4 = list(set(self.feature_set_3 + freq_features + cluster_features))

            self.feature_sets = {
                "set_1": self.feature_set_1,
                "set_2": self.feature_set_2,
                "set_3": self.feature_set_3,
                "set_4": self.feature_set_4
            }

            logging.info("Feature set preparation completed successfully.\n")
            return self.feature_sets
        except Exception as e:
            logging.error("ERROR: Failed to prepare feature sets.")
            raise MyException(e, sys)
        
    def perform_forward_selection(self):
        """
        Perform forward feature selection using the learner's method.

        Args:
            learner: An instance of ClassificationAlgorithms.
            X_train (pd.DataFrame): Training feature data.
            y_train (pd.Series): Training labels.
            max_features (int): Maximum number of features to select.

        Returns:
            tuple: (selected_features, ordered_features, ordered_scores)
        """
        try:
            self.learner = ClassificationAlgorithms()
            logging.info(f"Starting forward feature selection with max_features= {self.model_trainer_config.max_features}...")
            selected_features, ordered_features, ordered_scores = self.learner.forward_selection(self.model_trainer_config.max_features, self.X_train, self.y_train)
            
            logging.info("Feature selection completed successfully.")
            logging.info(f"Selected Features: {selected_features}")
            
            return selected_features, ordered_features, ordered_scores
        except Exception as e:
            logging.error("Error occurred during feature selection.")
            raise MyException(e, sys)
        
    def evaluate_feature_sets(self):
        """
        Trains multiple classifiers on different feature sets and evaluates their performance.
        
        Args:
            learner: An object containing various ML models as methods.
            X_train (pd.DataFrame): Training feature data.
            X_test (pd.DataFrame): Test feature data.
            y_train (pd.Series): Training labels.
            y_test (pd.Series): Test labels.
            possible_feature_sets (list): List of feature sets to evaluate.
            feature_names (list): Names corresponding to the feature sets.
            iterations (int, optional): Number of times to run non-deterministic classifiers. Default is 1.
        
        Returns:
            pd.DataFrame: A dataframe containing model performance scores.
        """
        self.learner = ClassificationAlgorithms()
        learner = self.learner
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        self.possible_feature_sets = [self.feature_set_1, self.feature_set_2, self.feature_set_3, self.feature_set_4, self.model_trainer_config.selected_features]        

        score_df = pd.DataFrame()
        
        for i, feature_set_name in enumerate(self.possible_feature_sets):
            print(f"Evaluating Feature Set {i}: {feature_set_name}")
            selected_train_X = X_train[self.possible_feature_sets[i]]
            selected_test_X = X_test[self.possible_feature_sets[i]]

            # Initialize performance metrics
            performance_test_nn, performance_test_rf = 0, 0

            # Train non-deterministic classifiers multiple times
            for it in range(ITERATIONS):
                print(f"\tTraining Neural Network, iteration {it+1}")
                _, class_test_y, _, _ = learner.feedforward_neural_network(
                    selected_train_X, y_train, selected_test_X, gridsearch=False
                )
                performance_test_nn += accuracy_score(y_test, class_test_y)

                print(f"\tTraining Random Forest, iteration {it+1}")
                _, class_test_y, _, _ = learner.random_forest(
                    selected_train_X, y_train, selected_test_X, gridsearch=True
                )
                performance_test_rf += accuracy_score(y_test, class_test_y)

            performance_test_nn /= ITERATIONS
            performance_test_rf /= ITERATIONS

            # Train deterministic classifiers
            print("\tTraining KNN")
            _, class_test_y, _, _ = self.learner.k_nearest_neighbor(
                selected_train_X, y_train, selected_test_X, gridsearch=True
            )
            performance_test_knn = accuracy_score(y_test, class_test_y)

            print("\tTraining Decision Tree")
            _, class_test_y, _, _ = self.learner.decision_tree(
                selected_train_X, y_train, selected_test_X, gridsearch=True
            )
            performance_test_dt = accuracy_score(y_test, class_test_y)

            print("\tTraining Naive Bayes")
            _, class_test_y, _, _ = self.learner.naive_bayes(
                selected_train_X, y_train, selected_test_X
            )
            performance_test_nb = accuracy_score(y_test, class_test_y)

            # Save results to dataframe
            models = ["NN", "RF", "KNN", "DT", "NB"]
            new_scores = pd.DataFrame(
                {
                    "model": models,
                    "feature_set": feature_set_name,
                    "accuracy": [
                        performance_test_nn,
                        performance_test_rf,
                        performance_test_knn,
                        performance_test_dt,
                        performance_test_nb,
                    ],
                }
            )
            score_df = pd.concat([score_df, new_scores], ignore_index=True)
            sorted_scores = score_df.sort_values(by="accuracy", ascending=False)
            logging.info(f"Sorted accuracy scores:\n{sorted_scores}")
        return score_df
    
    def train_and_evaluate_model(self):
        """
        Trains a given model on the selected feature set and evaluates its performance.

        Args:
            learner (object): The classification algorithms handler.
            model_name (str): The model to train (e.g., "random_forest", "knn").
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
            feature_set (list): Selected features for training.
            gridsearch (bool): Whether to use hyperparameter tuning.

        Returns:
            float: Accuracy score of the model.
            np.ndarray: Confusion matrix of the model.
        """
        try:
            learner = self.learner
            model_name = learner.random_forest()
            logging.info(f"Training {model_name} model with {len()} selected features...")

            model_function = getattr(learner, self.learner.random_forest)
            class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = model_function(
                self.X_train[self.possible_feature_sets], self.y_train, self.X_test[self.possible_feature_sets], gridsearch=True)

            accuracy = accuracy_score(self.y_test, class_test_y)
            classes = class_test_prob_y.columns
            cm = confusion_matrix(self.y_test, class_test_y, labels=classes)

            logging.info(f"{model_name} Model Accuracy: {accuracy:.4f}")
            logging.info(f"Confusion Matrix:\n{cm}")


            return accuracy, cm

        except Exception as e:
            logging.error(f"Error in training {model_name}: {str(e)}")
            raise e
        
    # def save_trained_model(self, model, save_path="./saved_models"):
    #     """
    #     Saves the trained model to a specified directory.

    #     Args:
    #         model (object): The trained model object.
    #         model_name (str): Name of the model (e.g., "random_forest").
    #         save_path (str): Directory path to save the model (default: "./saved_models").

    #     Returns:
    #         str: Path of the saved model file.
    #     """
    #     try:
    #         # Ensure the save directory exists
    #         os.makedirs(save_path, exist_ok=True)

    #         # Define the file path
    #         model_save_file = os.path.join(save_path, f"{self.model_name}.pkl")

    #         # Save the model
    #         joblib.dump(model, model_save_file)

    #         logging.info(f"Model '{self.model_name}' saved successfully at: {model_save_file}")
    #         return model_save_file

    #     except Exception as e:
    #         logging.error(f"Error saving model '{smodel_name}': {str(e)}")
    #         raise e


# Example Usage
# df = pd.read_csv("your_data.csv")
# features = ["participant", "category", "set"]
# target = "label"
# trainer = ModelTrainer(df, features, target)
# trainer.prepare_data()
# trainer.plot_distribution()
