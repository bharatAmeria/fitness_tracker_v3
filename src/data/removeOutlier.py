import pandas as pd
import numpy as np
import os
import sys
import wandb
import mlflow

from src.exception import MyException
from src.logger import logging
from src.entity.configEntity import RemoveOutlierConfig
from src.utils.outlierFunction import *
from src.constants import *

class RemoveOutlier:
    def __init__(self, outlier_removing_config: RemoveOutlierConfig, use_wandb=True):
        try:
            self.outlier_removing_config = outlier_removing_config
            self.use_wandb = use_wandb

            if self.use_wandb:
                wandb.init(project="Outlier_Removal", name="RemoveOutliers", config=vars(outlier_removing_config))
        except Exception as e:
            raise MyException(e, sys)

    def load_data(self):
        try:
            file_name = self.outlier_removing_config.root_dir
            self.df = pd.read_pickle(file_name)
            return self.df
        except Exception as e:
            raise MyException(e, sys)

    def remove_outliers(self, method='chauvenet'):
        try:
            df = self.load_data()
            cleaned_df = df.copy()
            outlier_columns = list(df.columns[:6])
            
            for col in outlier_columns:
                for label in df["label"].unique():
                    temp_df = df[df["label"] == label].copy()
                    try:
                        if method == 'iqr':
                            dataset = mark_outliers_iqr(temp_df, col)
                        elif method == 'chauvenet':
                            dataset = mark_outliers_chauvenet(temp_df, col)
                        elif method == 'lof':
                            dataset, _, _ = mark_outliers_lof(temp_df, [col])
                        else:
                            raise ValueError(f"Unsupported method: {method}")
                        
                        outlier_col = "outlier_lof" if method == 'lof' else f"{col}_outlier"
                        temp_df.loc[dataset[outlier_col], col] = np.nan
                        cleaned_df.loc[cleaned_df["label"] == label, col] = temp_df[col]

                        n_outliers = len(temp_df) - temp_df[col].count()
                        
                        if self.use_wandb:
                            wandb.log({f"{col}_outliers_removed": n_outliers})
                    except Exception:
                        continue
            
            return cleaned_df
        except Exception as e:
            raise MyException(e, sys)

    def export_data(self):
        try:
            if not hasattr(self, 'df'):
                raise AttributeError("Data not loaded. Please load data before exporting.")
            
            self.df.to_pickle(self.outlier_removing_config.outlier_removed_file_name)
            
            if self.use_wandb:
                wandb.save(self.outlier_removing_config.outlier_removed_file_name)
        except Exception as e:
            raise MyException(e, sys)
