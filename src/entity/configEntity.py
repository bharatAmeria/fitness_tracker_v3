import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP
    
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    root_dir: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME)
    source_URL: str = os.path.join(root_dir, URL)
    local_data_file: str = os.path.join(root_dir,  LOCAL_DATA_FILE)
    unzip_dir: str = os.path.join(root_dir, UNZIP_DIR)
