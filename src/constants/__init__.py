import os
from datetime import date
from dotenv import load_dotenv
from typing import Dict, List
import pandas as pd
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
URL: str = os.getenv("source_url")
LOCAL_DATA_FILE: str = "raw_data.zip"
RAW_DATA_INGESTION_DIR_NAME: str = "raw_data_ingestion"
RAW_DATA_INGESTION_INGESTED_DIR: str = "ingested"
UNZIP_DIR: str = "data"

"""
---------------------------------------------------------------
 Data Processing related constants start with PROCESS_DATA var name
---------------------------------------------------------------
"""
PROCESS_DATA_SAMPLING_CONFIG: Dict[str, str] = {
    'acc_x': "mean",
    'acc_y': "mean",
    'acc_z': "mean",
    'gyr_x': "mean",
    'gyr_y': "mean",
    'gyr_z': "mean",
    'participant': "last",
    'label': "last",
    'category': "last",
    'set': "last"
}
PROCESS_DATA_SENSOR_COLUMNS: List[str] = [
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "participant", "label", "category", "set"
]
PROCESS_DATA_DEFAULT_RESAMPLING_FREQUENCY: str = "200ms"
PROCESS_DATA_ACCELEROMETER_PATTERN: str = "Accelerometer"
PROCESS_DATA_GYROSCOPE_PATTERN: str = "Gyroscope"
PROCESS_DATA_INTERIM_DATA_DIR: str = "interim"
PROCESS_DATA_META_MOTION_DIR: str = "MetaMotion"
PROCESS_DATA_PROCESSED_FILE_NAME: str = "01_data_processed.pkl"

"""
---------------------------------------------------------------
 Outlier Removing related constants start with OUTLIER_ var name
---------------------------------------------------------------
"""
OUTLIER_PROCESSED_FILE_NAME: str = "01_data_processed.pkl"
OUTLIER_INTERIM_DATA_DIR:str = "interim"
OUTLIER_REMOVED_DATA_DIR = "outlier_removed"
OUTLIER_REMOVED_FILE_NAME = "02_outliers_removed_chauvenets.pkl"
OUTLIER_REPORTS = "outlier_reports"
REPORTS_PATH = "reports"

METHOD_IQR = "iqr"
METHOD_LOF = "lof"
METHOD_CHAUVENET = "chauvenet"
PREDICTOR_COLUMNS = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
CLUSTER_COLUMNS = ["acc_x", "acc_y", "acc_z"]
N_CLUSTERS=5
K_RANGE=(2, 10)
FIGURE_SIZE = (20, 5)
DPI: int = 100
LINE_WIDTH: int = 2
CUTTOFF: int =  1
FS = int(1000 / 200)
WS= int(2800 / 200)
FEATURES_EXTRACTED_FILE_NAME: str = "03_data_features.pkl"

"""
---------------------------------------------------------------
 Model Training related constants start with OUTLIER_ var name
---------------------------------------------------------------
"""
ITERATIONS: int = 1
MAX_FEATURES: int = 10

SELECTED_FEATURES = [
 "acc_z_freq_0.0_HZ_ws_14",
 "acc_x_freq_0.0_HZ_ws_14",
 "gyr_r_pse",
 "acc_y_freq_0.0_HZ_ws_14"
 "gyr_z_freq_0.714_HZ_ws_14",
 "gyr_r_freq_1.071_HZ_ws_14",
 "gyr_z_freq_0.357_HZ_ws_14",
 "gyr_x_freq_1.071_HZ_ws_14"
 "acc_x_max_freq_"
 "gyr_z_max_freq"
]