import pandas as pd
import numpy as np
import os
import sys

from src.exception import MyException
from src.logger import logging
from src.entity.configEntity import RemoveOutlierConfig
from src.utils.outlierFunction import *
from src.constants import *

class RemoveOutlier:
    """
    A class to handle outlier detection and removal from sensor data.

    Methods:
        load_data(): Loads data from a specified file.
        remove_outliers(method='chauvenet'): Removes outliers using a specified method.
        export_data(): Exports the processed DataFrame to a file.
    """

    def __init__(self, outlier_removing_config: RemoveOutlierConfig):
        """
        Initializes the RemoveOutlier class with the given configuration.

        Args:
            outlier_removing_config (RemoveOutlierConfig): Configuration object for outlier removal.
        """
        try:
            logging.info(f"\n{'='*40}\n   Outlier Removal Process Initiated   \n{'='*40}\n")
            logging.info("Initializing the RemoveOutlier class and loading configurations...")
            
            self.outlier_removing_config = outlier_removing_config
            
            logging.info("Configuration loaded successfully. Ready to process data.\n")
        except Exception as e:
            logging.error("\nERROR: Exception occurred during initialization of RemoveOutlier class.")
            raise MyException(e, sys)

    def load_data(self):
        """
        Loads data from the specified file.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            logging.info("\nLoading data from the specified file...")
            file_name = self.outlier_removing_config.root_dir
            self.df = pd.read_pickle(file_name)
            
            logging.info("Data loaded successfully.\n")
            return self.df
        except Exception as e:
            logging.error("\nERROR: Failed to load data.")
            raise MyException(e, sys)

    def remove_outliers(self, method='chauvenet'):
        """
        Removes outliers from the dataset using a specified method.

        Args:
            method (str, optional): Outlier detection method. Defaults to 'chauvenet'.
                - 'iqr': Uses Interquartile Range.
                - 'chauvenet': Uses Chauvenet's criterion.
                - 'lof': Uses Local Outlier Factor.

        Returns:
            pd.DataFrame: DataFrame with outliers removed.
        """
        try:
            logging.info(f"\n{'-'*40}\n   Outlier Removal Process Started   \n{'-'*40}\n")
            logging.info(f"Using '{method}' method for outlier detection.")
            
            df = self.load_data()
            cleaned_df = df.copy()
            outlier_columns = list(df.columns[:6])  # Selecting first six columns for outlier removal
            
            for col in outlier_columns:
                logging.info(f"\nProcessing column: '{col}' for outlier removal.")
                
                for label in df["label"].unique():
                    logging.info(f"\nProcessing label: '{label}' in column '{col}'.")
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

                        # Define outlier column name based on method
                        outlier_col = "outlier_lof" if method == 'lof' else f"{col}_outlier"
                        file_path = os.path.join(
                            self.outlier_removing_config.outlier_reports, 
                            f"{label.title()} ({outlier_col}) after removal.png"
                        )
                        
                        # Plot outliers and save the visualization
                        plot_binary_outliers(dataset, col, outlier_col, reset_index=True, save_path=file_path)
                        
                        # Replace outliers with NaN
                        temp_df.loc[dataset[outlier_col], col] = np.nan
                        cleaned_df.loc[cleaned_df["label"] == label, col] = temp_df[col]
                        
                        n_outliers = len(temp_df) - temp_df[col].count()
                        logging.info(f"Outliers removed from '{col}' for label '{label}': {n_outliers} outliers replaced with NaN.\n")
                    
                    except Exception as e:
                        logging.warning(f"WARNING: Error occurred while processing column '{col}' for label '{label}': {str(e)}\n")
                        continue
            
            logging.info(f"\n{'-'*40}\n   Outlier Removal Process Completed   \n{'-'*40}\n")
            return cleaned_df
        except Exception as e:
            logging.error("\nERROR: Exception occurred in remove_outliers method.")
            raise MyException(e, sys)

    def export_data(self):
        """
        Exports the processed DataFrame to a pickle file.
        """
        try:
            logging.info("\nStarting data export process...")
            
            # Ensure data is loaded before export
            if not hasattr(self, 'df'):
                raise AttributeError("Data not loaded. Please load data before exporting.")
            
            # Export DataFrame to pickle file
            self.df.to_pickle(self.outlier_removing_config.outlier_removed_file_name)
            
            logging.info(f"Data exported successfully to: {self.outlier_removing_config.outlier_removed_file_name}\n")
        except Exception as e:
            logging.error("\nERROR: Exception occurred in export_data method.")
            raise MyException(e, sys)
