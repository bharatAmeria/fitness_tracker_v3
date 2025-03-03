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
from src.components.data_processing.raw_data_ingestion import IngestData

load_dotenv()

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        
        Returns:
            DataIngestionArtifact: The artifact containing ingested data information
        """
        try:
            logging.info(f">>>>>> stage {os.getenv('STAGE1')} started <<<<<<\n")
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

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            logging.info("Starting the training pipeline.")

            data_ingestion_artifact = self.start_data_ingestion()
            logging.info("Data ingestion completed.")

            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)