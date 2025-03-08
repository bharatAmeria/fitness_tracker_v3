import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.configEntity import FeaturesExtractionConfig
from src.utils.dataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.utils.temporalAbstraction import NumericalAbstraction
from src.utils.frequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans
# from sklearn.impute import SimpleImputer


class FeaturesExtraction:
    """
    A class to handle the complete feature extraction process for sensor data.
    
    This includes:
    - Loading data
    - Handling missing values
    - Applying signal processing techniques
    - Performing dimensionality reduction (PCA)
    - Computing derived statistical features
    - Applying clustering

    The pipeline runs automatically when an instance is created.
    """

    def __init__(self, features_extraction_config: FeaturesExtractionConfig):
        """
        Initialize Feature Extraction with logging and error handling.
        Automatically executes the feature extraction pipeline.

        Args:
            features_extraction_config (FeaturesExtractionConfig): Configuration object containing file paths and parameters.
        """
        try:
            self.feature_config = features_extraction_config
            return

        except Exception as e:
            logging.error("Error in the Features Extraction pipeline")
            raise MyException(e, sys)

    def load_data(self):
        """
        Load sensor data from a pickle file.

        Returns:
            pd.DataFrame: Loaded sensor data.
        """
        try:
            file_name = self.feature_config.root_dir
            df = pd.read_pickle(file_name)
            self.predictor_columns = list(df.columns[:6])

            logging.info("Data loaded successfully")
            return df

        except Exception as e:
            logging.error("Error in loading data")
            raise MyException(e, sys)

    def handle_missing_values(self):
        """
        Handle missing values in sensor data by applying interpolation.

        Interpolation ensures that missing values are filled in a smooth manner
        without disrupting the data trends.
        """
        try:
            for col in self.predictor_columns:
                self.df[col] = self.df[col].interpolate()

            logging.info("Missing values handled successfully")
        except Exception as e:
            logging.error("Error in handling missing values")
            raise MyException(e, sys)

    def calculate_set_durations(self):
        """
        Calculate the duration of each recorded session/set in seconds.

        Adds a new 'duration' column to the dataframe.
        """
        try:
            for s in self.df["set"].unique():
                start = self.df[self.df["set"] == s].index[0]
                stop = self.df[self.df["set"] == s].index[-1]

                duration = stop - start
                self.df.loc[self.df["set"] == s, "duration"] = duration.seconds

            logging.info("Set durations calculated successfully")

        except Exception as e:
            logging.error("Error in calculating set durations")
            raise MyException(e, sys)

    def process_lowpass_filter(self):
        """
        Apply a low-pass Butterworth filter to remove high-frequency noise from sensor data.

        This smooths out signal fluctuations while preserving important trends.
        """
        try:
            Lowpass = LowPassFilter()
            for col in self.predictor_columns:
                self.df = Lowpass.low_pass_filter(self.df, col, FS, CUTTOFF, 5)
                self.df[col] = self.df[col + "_lowpass"]
                del self.df[col + "_lowpass"]

            logging.info("Low-pass filter applied successfully")

        except Exception as e:
            logging.error("Error in applying low-pass filter")
            raise MyException(e, sys)

    def process_pca(self):
        """
        Perform Principal Component Analysis (PCA) for dimensionality reduction.

        Reduces feature space while retaining maximum variance in the data.
        """
        try:
            PCA = PrincipalComponentAnalysis()
            self.df = PCA.apply_pca(self.df, self.predictor_columns, 3)

            logging.info("PCA applied successfully")

        except Exception as e:
            logging.error("Error in applying PCA")
            raise MyException(e, sys)

    def compute_sum_of_squares(self):
        """
        Compute sum of squares for acceleration and gyroscope readings.

        New columns:
        - 'acc_r': Magnitude of acceleration vector
        - 'gyr_r': Magnitude of gyroscope vector
        """
        try:
            self.df["acc_r"] = np.sqrt(
                self.df["acc_x"]**2 + self.df["acc_y"]**2 + self.df["acc_z"]**2
            )
            self.df["gyr_r"] = np.sqrt(
                self.df["gyr_x"]**2 + self.df["gyr_y"]**2 + self.df["gyr_z"]**2
            )

            logging.info("Sum of squares computed successfully")

        except Exception as e:
            logging.error("Error in computing sum of squares")
            raise MyException(e, sys)
        
    def process_features(self, sampling_rate=200):
        """
        Perform feature engineering: squaring features, temporal abstraction, and frequency feature extraction.
        
        Parameters:
        - df (pd.DataFrame): Input dataframe with sensor data.
        - predictor_columns (list): List of feature column names.
        - sampling_rate (int): Sampling rate in Hz (default: 200).
        
        Returns:
        - df_final (pd.DataFrame): Processed dataframe with additional features.
        """
        df_copy = self.df.copy()
        
        # Compute squared root of sum of squares (acceleration and gyroscope)
        df_copy["acc_r"] = np.sqrt(df_copy[["acc_x", "acc_y", "acc_z"]].pow(2).sum(axis=1))
        df_copy["gyr_r"] = np.sqrt(df_copy[["gyr_x", "gyr_y", "gyr_z"]].pow(2).sum(axis=1))
        
        # Temporal abstraction
        NumAbs = NumericalAbstraction()
        ws = int(1000 / sampling_rate)
        
        for col in self.predictor_columns + ["acc_r", "gyr_r"]:
            df_copy = NumAbs.abstract_numerical(df_copy, [col], ws, "mean")
            df_copy = NumAbs.abstract_numerical(df_copy, [col], ws, "std")
        
        df_temporal_list = []
        for s in df_copy["set"].unique():
            subset = df_copy[df_copy["set"] == s].copy()
            for col in self.predictor_columns:
                subset = NumAbs.abstract_numerical(subset, [col], ws, "mean")
                subset = NumAbs.abstract_numerical(subset, [col], ws, "std")
            df_temporal_list.append(subset)
        
        df_temporal = pd.concat(df_temporal_list)
        
        # Frequency feature extraction
        FreqAbs = FourierTransformation()
        fs = int(1000 / sampling_rate)
        ws = int(2800 / sampling_rate)
        
        df_freq_list = []
        for s in df_temporal["set"].unique():
            print(f"Applying Fourier Transformation to set {s}")
            subset = df_temporal[df_temporal["set"] == s].reset_index(drop=True).copy()
            subset = FreqAbs.abstract_frequency(subset, self.predictor_columns, ws, fs)
            df_freq_list.append(subset)

        df_final = pd.concat(df_freq_list)
        # df_final = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)
        return df_final


    def perform_clustering(self):
        """
        Perform clustering on the given DataFrame.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            cluster_columns (list): List of column names to be used for clustering.
            k_range (tuple): Range of k values to evaluate inertia.
            final_k (int): The number of clusters for the final clustering.
            
        Returns:
            pd.DataFrame: DataFrame with an added 'cluster' column.
        """
        cluster_columns = ["acc_x", "acc_y", "acc_z"]
        k_range=(2, 10)
        final_k=5
        # Preprocessing
        self.df_clean = self.df.dropna().iloc[::2].copy()
        
        # Determine optimal k using elbow method
        inertias = []
        k_values = range(*k_range)
        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
            cluster_labels = kmeans.fit_predict(self.df_clean[cluster_columns])
            inertias.append(kmeans.inertia_)
        
        # Final clustering
        kmeans = KMeans(n_clusters=final_k, n_init=20, random_state=0)
        self.df_clean["cluster"] = kmeans.fit_predict(self.df_clean[cluster_columns])
        
        return self.df_clean
    
    def export_data(self):
        """Export processed data to pickle file."""
        try:
            # Ensure df is loaded before export
            if not hasattr(self, 'df'):
                raise AttributeError("Data not loaded. Please load data before exporting.")

            os.makedirs(os.path.dirname(self.feature_config.features_extracted_file_name), exist_ok=True)

            self.df_clean.to_pickle(self.feature_config.features_extracted_file_name)  # Export the df attribute
            logging.info(f"Data exported successfully to {self.feature_config.features_extracted_file_name}")

        except Exception as e:
            logging.error("Error in export_data method")
            raise MyException(e, sys)
        
    def export_features(self):
        try:
            logging.info("\nStarting features export process...")  
            logging.info(f"{'='*20} Features Extraction Started {'='*20}")          
            self.df = self.load_data()

            self.handle_missing_values()
            self.calculate_set_durations()
            self.process_lowpass_filter()
            self.process_pca()
            self.compute_sum_of_squares()
            self.process_features()
            self.perform_clustering()
            self.export_data()

            logging.info(f"{'='*20} Features Extraction Completed {'='*20}")
        except Exception as e:
            logging.error("\nERROR: Exception occurred in export_data method.")
            raise MyException(e, sys)