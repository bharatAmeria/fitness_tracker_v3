import os
import sys
import pandas as pd
from glob import glob
import mlflow
from src.exception import MyException
from src.utils.main_utils import read_data
from src.constants import *
from src.entity.configEntity import MakeDatasetConfig

class MakeDataset:
    """
    Class for processing sensor data.
    This includes extracting metadata, processing accelerometer and gyroscope data,
    merging the datasets, and resampling the data for further analysis.
    """
    
    def __init__(self, make_dataset_config: MakeDatasetConfig):
        """
        Initializes the MakeDataset class.
        
        Args:
            make_dataset_config (MakeDatasetConfig): Configuration object containing paths for data processing.
        """
        try:
            self.config = make_dataset_config
            self.files = glob(os.path.join(self.config.raw_data_dir, "*.csv"))
            
            mlflow.log_param("raw_data_dir", self.config.raw_data_dir)
            mlflow.log_metric("num_files_found", len(self.files))
        except Exception as e:
            raise MyException(e, sys)

    def extract_features_from_filename(self, filepath: str) -> tuple:
        """
        Extracts metadata (participant, label, and category) from the filename.
        
        Args:
            filepath (str): Path to the CSV file.
        
        Returns:
            tuple: (participant, label, category) extracted from the filename.
        """
        try:
            filename = os.path.basename(filepath)
            parts = filename.split("-")
            participant, label, category = parts[0], parts[1], parts[2].rstrip("123").rstrip("_MetaWear_2019")
            return participant, label, category
        except Exception as e:
            raise MyException(e, sys)

    def process_sensor_data(self) -> tuple:
        """
        Reads and processes accelerometer and gyroscope sensor data.
        
        Returns:
            tuple: (DataFrame for accelerometer data, DataFrame for gyroscope data)
        """
        try:
            acc_df, gyr_df = pd.DataFrame(), pd.DataFrame()
            acc_set, gyr_set = 1, 1

            for file in self.files:
                participant, label, category = self.extract_features_from_filename(file)
                df = read_data(file)
                df["participant"], df["label"], df["category"] = participant, label, category

                if PROCESS_DATA_ACCELEROMETER_PATTERN in file:
                    df["set"] = acc_set
                    acc_set += 1
                    acc_df = pd.concat([acc_df, df])
                elif PROCESS_DATA_GYROSCOPE_PATTERN in file:
                    df["set"] = gyr_set
                    gyr_set += 1
                    gyr_df = pd.concat([gyr_df, df])

            for df in [acc_df, gyr_df]:
                if not df.empty and "epoch (ms)" in df.columns:
                    df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")
                    df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True, errors='ignore')

            mlflow.log_metric("num_accelerometer_records", len(acc_df))
            mlflow.log_metric("num_gyroscope_records", len(gyr_df))
            return acc_df, gyr_df
        except Exception as e:
            raise MyException(e, sys)

    def merge_sensor_data(self, acc_df: pd.DataFrame, gyr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges accelerometer and gyroscope data.
        
        Args:
            acc_df (pd.DataFrame): Accelerometer data.
            gyr_df (pd.DataFrame): Gyroscope data.
        
        Returns:
            pd.DataFrame: Merged sensor data.
        """
        try:
            data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
            data_merged.columns = PROCESS_DATA_SENSOR_COLUMNS
            mlflow.log_metric("merged_data_shape", data_merged.shape[0])
            return data_merged
        except Exception as e:
            raise MyException(e, sys)

    def resample_data(self, data_merged: pd.DataFrame, rule: str = PROCESS_DATA_DEFAULT_RESAMPLING_FREQUENCY) -> pd.DataFrame:
        """
        Resamples sensor data based on a given frequency.
        
        Args:
            data_merged (pd.DataFrame): Merged sensor data.
            rule (str): Resampling frequency (default: from constants file).
        
        Returns:
            pd.DataFrame: Resampled data.
        """
        try:
            days = [g for _, g in data_merged.groupby(pd.Grouper(freq="D"))]
            data_resampled = pd.concat([
                df.resample(rule=rule).apply(PROCESS_DATA_SAMPLING_CONFIG).dropna()
                for df in days
            ])
            data_resampled["set"] = data_resampled["set"].astype('int')
            mlflow.log_metric("resampled_data_shape", data_resampled.shape[0])
            return data_resampled
        except Exception as e:
            raise MyException(e, sys)
        
    def process_and_save(self):
        """
        Executes the full data processing pipeline and saves the processed dataset.
        
        Returns:
            pd.DataFrame: Final processed dataset.
        """
        try:
            if not self.files:
                return pd.DataFrame()

            acc_df, gyr_df = self.process_sensor_data()
            if acc_df.empty or gyr_df.empty:
                return pd.DataFrame()

            data_merged = self.merge_sensor_data(acc_df, gyr_df)
            data_resampled = self.resample_data(data_merged)

            os.makedirs(os.path.dirname(self.config.file_name), exist_ok=True)
            data_resampled.to_pickle(self.config.file_name)
            
            mlflow.log_param("processed_data_file", self.config.file_name)
            return data_resampled
        except Exception as e:
            raise MyException(e, sys)
