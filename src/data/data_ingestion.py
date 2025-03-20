import os
import sys
import zipfile
import gdown
import time  
import mlflow 

from src.exception import MyException
from src.logger import logging
from src.entity.configEntity import DataIngestionConfig

class IngestData:
    """
    Data ingestion class responsible for downloading and extracting data from a given source.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize the IngestData class.
        
        Args:
            data_ingestion_config (DataIngestionConfig): Configuration object containing source URL, file paths, etc.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def download_file(self) -> str:
        """
        Downloads a dataset from the provided Google Drive URL and saves it locally.
        
        Returns:
            str: Path to the downloaded file.
        
        Raises:
            MyException: If there is an error during download.
        """
        try:
            dataset_url = self.data_ingestion_config.source_URL
            zip_download_dir = os.path.dirname(self.data_ingestion_config.local_data_file)
            logging.info(f"Downloading data from {dataset_url}")
            logging.info(f"Downloading to: {self.data_ingestion_config.local_data_file}")

            os.makedirs(zip_download_dir, exist_ok=True)
            start_time = time.time()

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix + file_id, self.data_ingestion_config.local_data_file, quiet=False)

            end_time = time.time()
            download_time = end_time - start_time  
            file_size = os.path.getsize(self.data_ingestion_config.local_data_file) / (1024 * 1024)  

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            # ✅ Log download details to MLflow
            mlflow.log_metric("download_time_sec", download_time)
            mlflow.log_metric("file_size_MB", file_size)
            mlflow.log_param("dataset_url", dataset_url)

            return self.data_ingestion_config.local_data_file

        except Exception as e:
            raise MyException(e, sys)

    def extract_zip_file(self) -> None:
        """
        Extracts the downloaded zip file to the specified directory.
        
        Raises:
            MyException: If there is an error during extraction.
        """
        try:
            unzip_path = self.data_ingestion_config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            logging.info(f"Extracting from: {self.data_ingestion_config.local_data_file}")
            logging.info(f"Extracting to: {unzip_path}")

            if not os.path.isfile(self.data_ingestion_config.local_data_file):
                raise FileNotFoundError(f"Zip file not found at {self.data_ingestion_config.local_data_file}")

            start_time = time.time()

            with zipfile.ZipFile(self.data_ingestion_config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            end_time = time.time()
            extraction_time = end_time - start_time  

            logging.info(f"Successfully extracted zip file to {unzip_path}")

            # ✅ Log extraction details to MLflow
            mlflow.log_metric("extraction_time_sec", extraction_time)
            mlflow.log_param("extracted_path", unzip_path)

        except Exception as e:
            raise MyException(e, sys)