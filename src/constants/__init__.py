import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()  # 

"""
---------------------------------------------------------------
Training Pipeline related constant start with DATA_INGESTION VAR NAME
---------------------------------------------------------------
"""
# For training pipeline
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

"""
---------------------------------------------------------------
 Raw Data Ingestion related constant start with RAW_DATA VAR NAME
---------------------------------------------------------------
"""
# Raw data ingestion
URL = os.getenv("source_url")
LOCAL_DATA_FILE = "raw_data.zip"
RAW_DATA_INGESTION_DIR_NAME: str = "raw_data_ingestion"
RAW_DATA_INGESTION_INGESTED_DIR: str = "ingested"
UNZIP_DIR = "data"



