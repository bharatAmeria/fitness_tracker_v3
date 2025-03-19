""" Importing Python Libraries"""
import os
import sys
import zipfile
import gdown
import pandas as pd
import time  
import mlflow 
import wandb

""" Importing Modules"""
from typing import Optional
from src.exception import MyException
from src.logger import logging

""" Importing Classes"""
from src.entity.configEntity import DataIngestionConfig

class IngestData:
    """
    Data ingestion class which ingests data from the source and returns a DataFrame.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig, use_wandb: bool = False):
        """
        Initialize the data ingestion class.
        Args:
            data_ingestion_config: Configuration for data ingestion
            enable_wandb_logging: Flag to enable/disable WandB logging (default: False)
            enable_mlflow_logging: Flag to enable/disable MLflow logging (default: False)
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            self.enable_wandb_logging = use_wandb

            # ✅ Initialize WandB if enabled
            if self.enable_wandb_logging:
                import wandb
                wandb.init(project="data_ingestion", name="data_pipeline")  

        except Exception as e:
            raise MyException(e, sys)

    def download_file(self) -> str:
        """
        Fetch data from the URL and return the local file path.
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

            # ✅ Log to WandB if enabled
            if self.enable_wandb_logging:
                wandb.log({
                    "download_time_sec": download_time,
                    "file_size_MB": file_size,
                    "dataset_url": dataset_url
                })

            return self.data_ingestion_config.local_data_file

        except Exception as e:
            raise MyException(e, sys)

    def extract_zip_file(self) -> None:
        """
        Extract the downloaded zip file.
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

            # ✅ Log extraction time to WandB if enabled
            if self.enable_wandb_logging:
                wandb.log({
                    "extraction_time_sec": extraction_time,
                    "extracted_path": unzip_path
                })

        except Exception as e:
            raise MyException(e, sys)

