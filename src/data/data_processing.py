'''This file compiles multiple datasets into a single CSV file for further data processing.'''

""" Importing Python Libraries """
import os
import sys
import pandas as pd

""" Importing Modules """
from src.exception import MyException
from src.logger import logging
from glob import glob
from src.utils.main_utils import read_data

""" Importing Classes """
from src.constants import (
    PROCESS_DATA_SAMPLING_CONFIG,
    PROCESS_DATA_SENSOR_COLUMNS,
    PROCESS_DATA_DEFAULT_RESAMPLING_FREQUENCY,
    PROCESS_DATA_ACCELEROMETER_PATTERN,
    PROCESS_DATA_GYROSCOPE_PATTERN,
)
from src.entity.configEntity import MakeDatasetConfig


class MakeDataset:
    """
    Class for processing MetaMotion sensor data, including accelerometer and gyroscope readings.
    It compiles multiple dataset files, extracts relevant features, merges them, resamples them,
    and saves the final processed dataset.
    """

    def __init__(self, make_dataset_config: MakeDatasetConfig):
        """
        Initialize MakeDataset
        
        Args:
            make_dataset_config: Configuration for compiling the dataset
        """
        try:
            self.config = make_dataset_config
            self.files = glob(os.path.join(self.config.raw_data_dir, "*.csv"))
            
            logging.info("\n----------------------------------------")
            logging.info(f"Found {len(self.files)} files in {self.config.raw_data_dir}")
            logging.info("----------------------------------------\n")
        except Exception as e:
            raise MyException(e, sys)

    def extract_features_from_filename(self, filepath: str) -> tuple:
        """
        Extract metadata from filename, including participant ID, label, and category.
        
        Args:
            filepath: Path to the dataset file.
        
        Returns:
            tuple: (participant, label, category)
        """
        try:
            filename = os.path.basename(filepath)
            logging.info("\n----------------------------------------")
            logging.info(f"Extracting metadata from filename: {filename}")
            parts = filename.split("-")

            participant = parts[0]
            label = parts[1]
            category = parts[2].rstrip("123").rstrip("_MetaWear_2019")

            logging.info(f"Extracted -> Participant: {participant}, Label: {label}, Category: {category}")
            logging.info("----------------------------------------\n")
            return participant, label, category
        except Exception as e:
            raise MyException(e, sys)

    def process_sensor_data(self, acc_df=pd.DataFrame(), gyr_df=pd.DataFrame(), acc_set=1, gyr_set=1) -> tuple:
        """
        Process accelerometer and gyroscope data separately by reading files and appending metadata.
        
        Returns:
            tuple: (accelerometer_df, gyroscope_df) 
        """
        try:
            logging.info("\n----------------------------------------")
            logging.info("Processing sensor data files...")
            logging.info("----------------------------------------\n")
            
            for file in self.files:
                logging.info(f"Processing file: {file}\n")
                participant, label, category = self.extract_features_from_filename(file)
                df = read_data(file)

                df["participant"] = participant
                df["label"] = label
                df["category"] = category

                if PROCESS_DATA_ACCELEROMETER_PATTERN in file:
                    df["set"] = acc_set
                    acc_set += 1
                    acc_df = pd.concat([acc_df, df])
                    logging.info(f"Added {len(df)} accelerometer records\n")

                elif PROCESS_DATA_GYROSCOPE_PATTERN in file:
                    df["set"] = gyr_set
                    gyr_set += 1
                    gyr_df = pd.concat([gyr_df, df])
                    logging.info(f"Added {len(df)} gyroscope records\n")

            for df in [acc_df, gyr_df]:
                if not df.empty and "epoch (ms)" in df.columns:
                    df.index = pd.to_datetime(df["epoch (ms)"], unit="ms")
                    df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True, errors='ignore')

            logging.info(f"Processed {len(acc_df)} accelerometer and {len(gyr_df)} gyroscope readings\n")
            return acc_df, gyr_df
        except Exception as e:
            raise MyException(e, sys)

    def merge_sensor_data(self, acc_df: pd.DataFrame, gyr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge accelerometer and gyroscope data into a single dataset.
        
        Args:
            acc_df: Processed accelerometer data.
            gyr_df: Processed gyroscope data.
        
        Returns:
            pd.DataFrame: Merged sensor data.
        """
        try:
            logging.info("\n----------------------------------------")
            logging.info("Merging sensor data...")
            logging.info("----------------------------------------\n")
            
            data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
            data_merged.columns = PROCESS_DATA_SENSOR_COLUMNS
            
            logging.info(f"Merged dataset shape: {data_merged.shape}\n")
            return data_merged
        except Exception as e:
            raise MyException(e, sys)

    def resample_data(self, data_merged: pd.DataFrame, rule: str = PROCESS_DATA_DEFAULT_RESAMPLING_FREQUENCY) -> pd.DataFrame:
        """
        Resample data to a specified frequency.
        
        Args:
            data_merged: Merged sensor data.
            rule: Resampling frequency (default: "200ms").
        
        Returns:
            pd.DataFrame: Resampled data.
        """
        try:
            logging.info("\n----------------------------------------")
            logging.info(f"Resampling data to {rule} frequency...")
            logging.info("----------------------------------------\n")
            
            days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]
            data_resampled = pd.concat([
                df.resample(rule=rule)
                .apply(PROCESS_DATA_SAMPLING_CONFIG)
                .dropna()
                for df in days
            ])
            data_resampled["set"] = data_resampled["set"].astype('int')
            
            logging.info(f"Resampled dataset shape: {data_resampled.shape}\n")
            return data_resampled
        except Exception as e:
            raise MyException(e, sys)
        
    def process_and_save(self):
        """Process all data and save to a pickle file."""
        try:
            if not self.files:
                logging.warning("No files found in the specified directory.")
                return pd.DataFrame()

            acc_df, gyr_df = self.process_sensor_data()

            if acc_df.empty or gyr_df.empty:
                logging.warning("Processed data is empty. Skipping saving.")
                return pd.DataFrame()

            data_merged = self.merge_sensor_data(acc_df, gyr_df)
            data_resampled = self.resample_data(data_merged)

            os.makedirs(os.path.dirname(self.config.file_name), exist_ok=True)

            data_resampled.to_pickle(self.config.file_name)
            logging.info(f"Successfully saved processed data to {self.config.file_name}")

            return data_resampled

        except Exception as e:
            raise MyException(e, sys)
