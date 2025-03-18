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
            df_train = self.df[["participant", "category", "set", "label"]].copy()
            X = df_train.drop(columns=["label"])
            y = df_train["label"]

            # Splitting the dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42, stratify=y
            )

            logging.info("Train-test split completed successfully.\n")
      
            return X_train, X_test, y_train, y_test, df_train
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

            # Get df_train from split_data_as_train_test
            _, _, _, _, df_train = self.split_data_as_train_test()


            if self.df is None:
                logging.info("Data is not loaded. Loading data first...")
                self.load_data()

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

            feature_sets = [
                feature_set_1,
                feature_set_2,
                feature_set_3,
                feature_set_4
            ]

            logging.info("Feature set preparation completed successfully.\n")
            return feature_set_1, feature_set_2, feature_set_3, feature_set_4, feature_sets
        except Exception as e:
            logging.error("ERROR: Failed to prepare feature sets.")
            raise MyException(e, sys)
        
    # def evaluate_feature_sets(self):
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
        # Get X_train, y_train from split_data_as_train_test
        X_train, X_test, y_train, y_test, _ = self.split_data_as_train_test()
        feature_set_1, feature_set_2, feature_set_3, feature_set_4, _ = self.evaluate_feature_sets()

        learner = ClassificationAlgorithms()

        # Ensure possible feature sets are lists
        possible_feature_sets = [
            list(feature_set_1),
            list(feature_set_2),
            list(feature_set_3),
            list(feature_set_4),
            list(self.model_trainer_config.selected_features),
        ]

        print(f"Possible Feature Sets: {len(possible_feature_sets)} sets")

        for i, feature_set in enumerate(possible_feature_sets):
            print(f"Feature Set {i+1} ({len(feature_set)} features):\n{feature_set}\n")

        score_df = pd.DataFrame()
        
        for i, feature_set_name in enumerate(possible_feature_sets):
            print(f"Evaluating Feature Set {i}: {feature_set_name}")
            selected_train_X = X_train[possible_feature_sets[i]]
            selected_test_X = X_test[possible_feature_sets[i]]

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
            _, class_test_y, _, _ = learner.decision_tree(
                selected_train_X, y_train, selected_test_X, gridsearch=True
            )
            performance_test_dt = accuracy_score(y_test, class_test_y)

            print("\tTraining Naive Bayes")
            _, class_test_y, _, _ = learner.naive_bayes(
                selected_train_X, y_train, selected_test_X
            )
            performance_test_nb = accuracy_score(y_test, class_test_y)

            # Save results to dataframe
            models = ["NN", "RF", "KNN", "DT", "NB"]

            # Initialize variables to avoid undefined issues
            performance_test_nn = None
            performance_test_rf = None
            performance_test_knn = None
            performance_test_dt = None
            performance_test_nb = None
            
            # Your model training code that sets these variables...
            
            # Before creating the DataFrame, verify all variables are defined
            models = []
            accuracies = []
            
            # Only add models that have been successfully trained
            if performance_test_nn is not None:
                models.append("NN")
                accuracies.append(performance_test_nn)
            if performance_test_rf is not None:
                models.append("RF")
                accuracies.append(performance_test_rf)
            if performance_test_knn is not None:
                models.append("KNN")
                accuracies.append(performance_test_knn)
            if performance_test_dt is not None:
                models.append("DT")
                accuracies.append(performance_test_dt)
            if performance_test_nb is not None:
                models.append("NB")
                accuracies.append(performance_test_nb)
                
            # Now create the DataFrame with verified data
            if models and accuracies:  # Make sure we have data to add
                new_scores = pd.DataFrame(
                    {
                        "model": models,
                        "feature_set": [feature_set_name] * len(models),
                        "accuracy": accuracies,
                    }
                )
                score_df = pd.concat([score_df, new_scores], ignore_index=True)
                sorted_scores = score_df.sort_values(by="accuracy", ascending=False)
                logging.info(f"Sorted accuracy scores:\n{sorted_scores}")
            else:
                logging.warning("No models were successfully trained for this feature set")
                
            return score_df
            
        #     new_scores = pd.DataFrame(
        #         {
        #             "model": models,
        #             "feature_set": feature_set_name,
        #             "accuracy": [
        #                 performance_test_nn,
        #                 performance_test_rf,
        #                 performance_test_knn,
        #                 performance_test_dt,
        #                 performance_test_nb,
        #             ],
        #         }
        #     )
        #     score_df = pd.concat([score_df, new_scores], ignore_index=True)
        #     sorted_scores = score_df.sort_values(by="accuracy", ascending=False)
        #     logging.info(f"Sorted accuracy scores:\n{sorted_scores}")
        # return score_df
    
    # def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
            """
            Train the model and evaluate its performance
            
            Args:
                X_train (pd.DataFrame): Training features
                y_train (pd.Series): Training labels
                X_test (pd.DataFrame): Testing features
                y_test (pd.Series): Testing labels
                
            Returns:
                dict: Dictionary containing model performance metrics
                
            Raises:
                MyException: If training or evaluation fails
            """
            try:
                logging.info("Started training the model...")
                
                # Check for and handle non-numeric data in features
                for column in X_train.columns:
                    if X_train[column].dtype == 'object':
                        logging.warning(f"Column '{column}' contains string values. Attempting to convert.")
                        try:
                            # Try to convert directly if possible (for values like '1.5')
                            X_train[column] = pd.to_numeric(X_train[column], errors='coerce')
                            X_test[column] = pd.to_numeric(X_test[column], errors='coerce')
                            
                            # Fill NaN values that couldn't be converted
                            X_train[column].fillna(X_train[column].mean() if not pd.isna(X_train[column].mean()) else 0, inplace=True)
                            X_test[column].fillna(X_test[column].mean() if not pd.isna(X_test[column].mean()) else 0, inplace=True)
                        except:
                            # For categorical data, use one-hot encoding
                            logging.info(f"Converting categorical column '{column}' using one-hot encoding")
                            # Combine train and test to ensure consistent encoding
                            all_data = pd.concat([X_train[column], X_test[column]], axis=0)
                            dummies = pd.get_dummies(all_data, prefix=column, drop_first=True)
                            
                            # Split back into train and test
                            train_dummies = dummies.iloc[:len(X_train)]
                            test_dummies = dummies.iloc[len(X_train):]
                            
                            # Drop original column and add new dummy columns
                            X_train = X_train.drop(column, axis=1)
                            X_test = X_test.drop(column, axis=1)
                            X_train = pd.concat([X_train, train_dummies], axis=1)
                            X_test = pd.concat([X_test, test_dummies], axis=1)
                
                # Train the model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_train_pred = self.model.predict(X_train)
                y_test_pred = self.model.predict(X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                # Generate classification report with proper error handling
                try:
                    test_report = classification_report(y_test, y_test_pred)
                except Exception as report_error:
                    logging.warning(f"Could not generate classification report: {report_error}")
                    test_report = "Classification report unavailable"
                
                metrics = {
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'classification_report': test_report
                }
                
                logging.info(f"Model training completed. Test accuracy: {test_accuracy:.4f}")
                
                return metrics
                
            except Exception as e:
                logging.error(f"Error occurred during model training: {str(e)}")
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