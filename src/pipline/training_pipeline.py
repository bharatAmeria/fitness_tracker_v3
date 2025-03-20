import os
import sys
import mlflow
from dotenv import load_dotenv

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.configEntity import *
from src.data.data_ingestion import IngestData
from src.data.data_processing import MakeDataset
from src.data.removeOutlier import RemoveOutlier
from src.features.buildFeatures import FeaturesExtraction
from src.models.modelTrainer import ModelTrainer

load_dotenv()

class TrainPipeline:
    """
    This class defines the end-to-end training pipeline, which includes data ingestion, preprocessing,
    outlier removal, feature engineering, and model training.
    """
    def __init__(self):
        """
        Initializes the TrainPipeline with configurations for each stage of the ML pipeline.
        """
        self.data_ingestion_config = DataIngestionConfig()
        self.data_processing_config = MakeDatasetConfig()
        self.outlier_removing_config = RemoveOutlierConfig()
        self.features_config = FeaturesExtractionConfig()
        self.model_config = ModelTrainerConfig()

    def start_data_ingestion(self):
        """
        Executes the data ingestion process, including downloading and extracting data.
        """
        try:
            logging.info("Starting data ingestion.")
            
            data_ingestion = IngestData(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            
            logging.info("Data ingestion completed.")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_processing(self):
        """
        Processes the ingested data to prepare it for further analysis.
        """
        try:
            logging.info("Starting data processing.")
            
            data_processor = MakeDataset(make_dataset_config=self.data_processing_config)
            processed_data = data_processor.process_and_save()
            
            logging.info("Data processing completed.")
            return processed_data
        except Exception as e:
            raise MyException(e, sys)

    def start_removing_outliers(self):
        """
        Removes outliers from the dataset using the specified method.
        """
        try:
            logging.info("Removing outliers from the dataset.")
            
            outlier = RemoveOutlier(outlier_removing_config=self.outlier_removing_config, use_wandb=False)
            outlier_df = outlier.remove_outliers(method=METHOD_CHAUVENET)
            outlier.export_data()
            
            logging.info("Outlier removal completed.")
            return outlier_df
        except Exception as e:
            raise MyException(e, sys)

    def start_features_engg(self):
        """
        Extracts relevant features from the dataset.
        """
        try:
            logging.info("Extracting features.")
            
            feature = FeaturesExtraction(features_extraction_config=self.features_config)
            export_features = feature.export_features()
            
            logging.info("Feature extraction completed.")
            return export_features
        except Exception as e:
            raise MyException(e, sys)

    def start_model_training(self):
        """
        Trains the machine learning model using the preprocessed data.
        """
        try:
            logging.info("Starting model training.")
            
            trainer = ModelTrainer(model_trainer_config=self.model_config)
            trainer.load_data()
            trainer.split_data_as_train_test()
            trainer.prepare_feature_set()
            trainer.grid_search_model_selection()
            
            logging.info("Model training completed.")
            return
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        """
        Executes the complete ML pipeline sequentially.
        """
        try:
            logging.info("Running the training pipeline.")
            
            with mlflow.start_run():
                self.start_data_ingestion()
                self.start_data_processing()
                self.start_removing_outliers()
                self.start_features_engg()
                self.start_model_training()
                
                logging.info("Training pipeline completed successfully.")
        except Exception as e:
            raise MyException(e, sys)

