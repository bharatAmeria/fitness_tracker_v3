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
    max_features: int = MAX_FEATURES
    selected_features = SELECTED_FEATURES
    # model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    # trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME)
    # expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
    # model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    # _n_estimators = MODEL_TRAINER_N_ESTIMATORS
    # _min_samples_split = MODEL_TRAINER_MIN_SAMPLES_SPLIT
    # _min_samples_leaf = MODEL_TRAINER_MIN_SAMPLES_LEAF
    # _max_depth = MIN_SAMPLES_SPLIT_MAX_DEPTH
    # _criterion = MIN_SAMPLES_SPLIT_CRITERION
    # _random_state = MIN_SAMPLES_SPLIT_RANDOM_STATE
