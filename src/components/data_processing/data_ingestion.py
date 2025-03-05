""" Impoting Python Libraries"""
import os
import sys
import zipfile
import gdown
import pandas as pd

""" Importing Modules"""
from typing import Optional
from src.exception import MyException
from src.logger import logging

""" Importing Classes"""
from src.entity.configEntity import DataIngestionConfig
from src.entity.artifactEntity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging


class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize the data ingestion class.
        Args:
            data_ingestion_config: Configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def download_file(self) -> str:
        """
        Fetch data from the url and return DataIngestionArtifact
        Returns:
            DataIngestionArtifact: Contains paths to downloaded and extracted data
        """
        try:
            dataset_url = self.data_ingestion_config.source_URL
            zip_download_dir = os.path.dirname(self.data_ingestion_config.local_data_file)
            logging.info(f"Downloading data from {self.data_ingestion_config.source_URL}")
            logging.info(f"Downloading to: {self.data_ingestion_config.local_data_file}")

            # Download logic here
            download_dir = os.path.dirname(self.data_ingestion_config.local_data_file)
            os.makedirs(download_dir, exist_ok=True)
            logging.info(f"Downloading file to: {self.data_ingestion_config.local_data_file}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, self.data_ingestion_config.local_data_file, quiet=False)

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return self.data_ingestion_config.local_data_file

        except Exception as e:
            raise MyException(e, sys)

    def extract_zip_file(self) -> None:
        """
        Extract the downloaded zip file.

        Returns:
            None

        Raises:
            MyException: If extraction fails
        """
        try:
            unzip_path = self.data_ingestion_config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            # Add debug logging
            logging.info(f"Attempting to extract from: {self.data_ingestion_config.local_data_file}")
            logging.info(f"Extracting to path: {unzip_path}")

            # Check if source file exists
            if not os.path.isfile(self.data_ingestion_config.local_data_file):
                raise FileNotFoundError(f"Zip file not found at {self.data_ingestion_config.local_data_file}")

            # Fix: Use local_data_file instead of unzip_dir for the source zip file
            with zipfile.ZipFile(self.data_ingestion_config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logging.info(f"Extracted zip file to {unzip_path}")
            logging.info(f"Successfully extracted zip file to {unzip_path}")

        except Exception as e:
            raise MyException(e, sys)