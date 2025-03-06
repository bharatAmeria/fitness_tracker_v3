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

load_dotenv()

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_processing_config = MakeDatasetConfig()

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
            logging.info(f"\n>>>>>> stage {os.getenv('STAGE2')} started <<<<<<")
            return data_processing_artifact, processed_data

        except Exception as e:
            raise MyException(e, sys)


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

            return data_ingestion_artifact, data_processing_artifact
        except Exception as e:
            raise MyException(e, sys)