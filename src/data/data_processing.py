import os
import sys
import pandas as pd
import wandb
from glob import glob
from src.exception import MyException
from src.utils.main_utils import read_data
from src.constants import *
from src.entity.configEntity import MakeDatasetConfig

class MakeDataset:
    def __init__(self, make_dataset_config: MakeDatasetConfig, use_wandb: bool =True):
        try:
            self.config = make_dataset_config
            self.files = glob(os.path.join(self.config.raw_data_dir, "*.csv"))
            self.use_wandb = use_wandb
            wandb.init(project="sensor_data_processing", job_type="data_preprocessing")
            wandb.config.update({"raw_data_dir": self.config.raw_data_dir})
            wandb.log({"num_files_found": len(self.files)})
        except Exception as e:
            raise MyException(e, sys)

    def extract_features_from_filename(self, filepath: str) -> tuple:
        try:
            filename = os.path.basename(filepath)
            parts = filename.split("-")
            participant, label, category = parts[0], parts[1], parts[2].rstrip("123").rstrip("_MetaWear_2019")
            return participant, label, category
        except Exception as e:
            raise MyException(e, sys)

    def process_sensor_data(self) -> tuple:
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

            wandb.log({"num_accelerometer_records": len(acc_df), "num_gyroscope_records": len(gyr_df)})
            return acc_df, gyr_df
        except Exception as e:
            raise MyException(e, sys)

    def merge_sensor_data(self, acc_df: pd.DataFrame, gyr_df: pd.DataFrame) -> pd.DataFrame:
        try:
            data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
            data_merged.columns = PROCESS_DATA_SENSOR_COLUMNS
            wandb.log({"merged_data_shape": data_merged.shape})
            return data_merged
        except Exception as e:
            raise MyException(e, sys)

    def resample_data(self, data_merged: pd.DataFrame, rule: str = PROCESS_DATA_DEFAULT_RESAMPLING_FREQUENCY) -> pd.DataFrame:
        try:
            days = [g for _, g in data_merged.groupby(pd.Grouper(freq="D"))]
            data_resampled = pd.concat([
                df.resample(rule=rule).apply(PROCESS_DATA_SAMPLING_CONFIG).dropna()
                for df in days
            ])
            data_resampled["set"] = data_resampled["set"].astype('int')
            wandb.log({"resampled_data_shape": data_resampled.shape})
            return data_resampled
        except Exception as e:
            raise MyException(e, sys)
        
    def process_and_save(self):
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
            wandb.save(self.config.file_name)
            return data_resampled
        except Exception as e:
            raise MyException(e, sys)

