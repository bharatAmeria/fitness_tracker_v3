""" Impoting Python Libraries """
import os
import sys

""" Importing Modules """
from src.exception import MyException
from src.logger import logging as logging

""" Importing Classes """
from src.constants import *
from src.entity.artifactEntity import *
from src.entity.configEntity import *
from src.data.data_ingestion import IngestData
from src.data.data_processing import MakeDataset
from src.data.removeOutlier import RemoveOutlier
from src.features.buildFeatures import FeaturesExtraction
from src.models.modelTrainer import ModelTrainer
load_dotenv()

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_processing_config = MakeDatasetConfig()
        self.outlier_removing_config = RemoveOutlierConfig()
        self.features_config = FeaturesExtractionConfig()
        self.model_config = ModelTrainerConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        
        Returns:
            DataIngestionArtifact: The artifact containing ingested data information
        """
        try:
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE1')} started <<<<<<")
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")

            logging.info("Getting the raw_data from google drive")
            data_ingestion = IngestData(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.download_file()
            data_ingestion.extract_zip_file()

            logging.info("Got the raw_data file in the artifact folder")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            logging.info(f">>>>>> stage {os.getenv('STAGE1')} ended <<<<<< \n ")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)
        
    def start_data_processing(self) -> DataProcessingArtifact:
        """
        This method processes the raw sensor data and saves the processed data.

        Returns:
            DataProcessingArtifact: The artifact containing processed data information
        """
        try:
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE2')} started <<<<<<")
            logging.info("Entered the start_data_processing method of TrainPipeline class")

            logging.info("Initializing data processing")
            data_processor = MakeDataset(make_dataset_config=self.data_processing_config)

            logging.info("Starting data processing pipeline")
            processed_data = data_processor.process_and_save()

            # Create a DataProcessingArtifact
            data_processing_artifact = DataProcessingArtifact(
                processed_file_path=data_processor.config.interim_dataset_dir,
                is_processed=True,
                message="Data processing completed successfully"
            )
            # data_processing_artifact = data_processing_artifact.

            logging.info(f"Data processing completed. Processed data saved at: {data_processor.config.interim_dataset_dir}")
            logging.info("Exited the start_data_processing method of TrainPipeline class")
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE2')} ended <<<<<<")
            return data_processing_artifact, processed_data

        except Exception as e:
            raise MyException(e, sys)
        
    def start_removing_outliers(self):
        try:
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE3')} started <<<<<<")
            logging.info("Entered the start_removing_outliers method of TrainPipeline class")

            logging.info("Entered the start_removing_outliers method of training pipeline")
            logging.info("Initializing outlier removing pipeline.")
            outlier = RemoveOutlier(outlier_removing_config=self.outlier_removing_config)

            logging.info("outlier after removal")
            outlier_df = outlier.remove_outliers(method=METHOD_CHAUVENET)
            outlier.export_data()
            logging.info("Exited the start_removing_outliers method of TrainPipeline class")
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE3')} ended <<<<<<")

            return outlier_df
        except Exception as e:
            logging.error("Error in removing outliers: %s", str(e))
            raise

    def start_features_engg(self):
        try:
            logging.info("Entered the start_feature_engg method of training pipeline")
            logging.info("Initializing feature_engg pipeline.")
            feature = FeaturesExtraction(features_extraction_config=self.features_config)
            export_features = feature.export_features()

            return export_features
        except Exception as e:
            logging.error("Error in removing outliers: %s", str(e))
            raise

    def start_model_training(self):
        try:
            trainer = ModelTrainer(model_trainer_config=self.model_config)
            data = trainer.load_data()
            split_data = trainer.split_data_as_train_test()
            prepare_features = trainer.prepare_feature_set() 
            # forward_pass = trainer.perform_forward_selection()
            # evaluate = trainer.evaluate_feature_sets()
            # train  = trainer.train()

            return data, prepare_features, split_data

        except Exception as e:
            logging.error("Error in Model Training: %s", str(e))
            raise


    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            logging.info("Starting the training pipeline.")

            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed.")

            data_processing_artifact = self.start_data_processing()
            logging.info("Data processing completed.")

            removing_outlier = self.start_removing_outliers()
            logging.info("Outlier Removed from the data successfully.")

            featuresExtract = self.start_features_engg()
            logging.info("Features Extareted Successfully Completed")

            modelTraining = self.start_model_training()
            logging.info("Model Training Successfully Completed")
            
            logging.info("Training Pipeline Successfully Completed")
            return data_ingestion_artifact, data_processing_artifact, removing_outlier, featuresExtract, modelTraining
        except Exception as e:
            raise MyException(e, sys)