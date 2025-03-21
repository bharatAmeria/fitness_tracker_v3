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
    source_URL: str = "https://drive.google.com/file/d/11AXpnvTdbz4J7DElNtDVjVE3mvPkwDzA/view?usp=sharing"
    local_data_file: str = os.path.join(root_dir,  LOCAL_DATA_FILE)
    unzip_dir: str = os.path.join(root_dir, UNZIP_DIR)

@dataclass
class MakeDatasetConfig:
    """Configuration for compiling the dataset."""

    root_dir: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, UNZIP_DIR, PROCESS_DATA_META_MOTION_DIR)
    raw_data_dir: str = root_dir
    interim_dataset_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                            RAW_DATA_INGESTION_DIR_NAME,
                                            PROCESS_DATA_INTERIM_DATA_DIR,
                                            )
    file_name: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, PROCESS_DATA_INTERIM_DATA_DIR, PROCESS_DATA_PROCESSED_FILE_NAME)

@dataclass
class RemoveOutlierConfig:
    """Configuration for outlier removing"""
    root_dir = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, OUTLIER_INTERIM_DATA_DIR, OUTLIER_PROCESSED_FILE_NAME)
    outlier_removed_file_name = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, OUTLIER_INTERIM_DATA_DIR, OUTLIER_REMOVED_FILE_NAME)
    outlier_reports = os.path.join(training_pipeline_config.artifact_dir, REPORTS_PATH, OUTLIER_REPORTS)

@dataclass
class FeaturesExtractionConfig:
    """Configuration for features extraction"""
    root_dir = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, OUTLIER_INTERIM_DATA_DIR, OUTLIER_PROCESSED_FILE_NAME)
    features_extracted_file_name = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, OUTLIER_INTERIM_DATA_DIR, FEATURES_EXTRACTED_FILE_NAME)

@dataclass
class ModelTrainerConfig:
    data: str = os.path.join(training_pipeline_config.artifact_dir, RAW_DATA_INGESTION_DIR_NAME, OUTLIER_INTERIM_DATA_DIR, FEATURES_EXTRACTED_FILE_NAME)
    model: str = os.path.join(MODEL_TRAINER_DIR_NAME)
    max_features: int = MAX_FEATURES
