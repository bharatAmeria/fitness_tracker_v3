import os
import sys
import pandas as pd
import numpy as np

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.configEntity import FeaturesExtractionConfig
from src.utils.dataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.utils.temporalAbstraction import NumericalAbstraction
from src.utils.frequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

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
            df = self.load_data()
            df_missing = df
            predictor_columns = PREDICTOR_COLUMNS

            for col in predictor_columns:
                df_missing[col] = df_missing[col].interpolate()

            logging.info("Missing values handled successfully")
            return df_missing
        except Exception as e:
            logging.error("Error in handling missing values")
            raise MyException(e, sys)

    def calculate_set_durations(self):
        """
        Calculate the duration of each recorded session/set in seconds.

        Adds a new 'duration' column to the dataframe.
        """
        try:
            df = self.handle_missing_values()
            df_set = df
            for s in df_set["set"].unique():
                start = df_set[df_set["set"] == s].index[0]
                stop = df_set[df_set["set"] == s].index[-1]

                duration = stop - start
                df_set.loc[df_set["set"] == s, "duration"] = duration.seconds

            
            logging.info("Set durations calculated successfully")
            return df_set
        except Exception as e:
            logging.error("Error in calculating set durations")
            raise MyException(e, sys)

    def process_lowpass_filter(self):
        """
        Apply a low-pass Butterworth filter to remove high-frequency noise from sensor data.

        This smooths out signal fluctuations while preserving important trends.
        """
        try:
            df = self.calculate_set_durations()
            df_lowpass = df

            Lowpass = LowPassFilter()
            for col in PREDICTOR_COLUMNS:
                df_lowpass = Lowpass.low_pass_filter(df_lowpass, col, FS, CUTTOFF, order=5)
                df_lowpass[col] = df_lowpass[col + "_lowpass"]
                del df_lowpass[col + "_lowpass"]

            logging.info("Low-pass filter applied successfully")

            return df_lowpass
        except Exception as e:
            logging.error("Error in applying low-pass filter")
            raise MyException(e, sys)

    def perform_pca(self):
        """
        Perform PCA on the given DataFrame and return the transformed DataFrame.

        Parameters:
        - df_lowpass (pd.DataFrame): Filtered input DataFrame.
        - predictor_columns (list): List of feature column names for PCA.
        - num_components (int): Number of principal components to retain.

        Returns:
        - df_pca (pd.DataFrame): DataFrame with PCA components.
        - pc_values (list): Explained variance for each principal component.
        """
        df = self.process_lowpass_filter()
        df_pca = df
        PCA = PrincipalComponentAnalysis()

        # Determine explained variance
        pc_values = PCA.determine_pc_explained_variance(df_pca, PREDICTOR_COLUMNS)

        # Apply PCA
        df_pca = PCA.apply_pca(df_pca, PREDICTOR_COLUMNS, 3)

        return df_pca
    
    def compute_sum_of_squares(self):
        """
        Compute sum of squares for acceleration and gyroscope data.
        """
        df_pca = self.perform_pca()
        df_squared = df_pca
        df_squared["acc_r"] = np.sqrt(df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2)
        df_squared["gyr_r"] = np.sqrt(df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2)
        return df_squared
    
    def apply_temporal_abstraction(self):
        """
        Apply temporal abstraction using mean and standard deviation over a sliding window.
        """
        df = self.compute_sum_of_squares()
        df_temporal = df

        NumAbs = NumericalAbstraction()
        predictor_columns = PREDICTOR_COLUMNS + ["acc_r", "gyr_r"]
        
        for col in predictor_columns:
            df_temporal = NumAbs.abstract_numerical(df_temporal, [col], WS, "mean")
            df_temporal = NumAbs.abstract_numerical(df_temporal, [col], WS, "std")
        
        df_temporal_list = []
        for s in df_temporal["set"].unique():
            subset = df_temporal[df_temporal["set"] == s].copy()
            for col in predictor_columns:
                subset = NumAbs.abstract_numerical(subset, [col], WS, "mean")
                subset = NumAbs.abstract_numerical(subset, [col], WS, "std")
            df_temporal_list.append(subset)
        
            temporal = pd.concat(df_temporal_list)


        
        df_freq = temporal.copy().reset_index()
        FreqAbs = FourierTransformation()

        df_freq = FreqAbs.abstract_frequency(temporal, ["acc_y"], WS, FS)
        df_temporal_list = []
        
        for s in temporal["set"].unique():
            logging.info(f"Applying Fourier Transformation to set {s}")
            subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
            subset = FreqAbs.abstract_frequency(subset, predictor_columns, WS, FS)
            df_temporal_list.append(subset)
        
        df_freq = pd.concat(df_temporal_list)

        # Drop NaN values and reduce data size
        df_freq = df_freq.dropna()
        df_freq = df_freq.iloc[::2]

        return df_freq
    
    def process_clustering(self):
        """
        Process overlapping windows, perform clustering on accelerometer data,
        and return the DataFrame with cluster labels.
        
        Parameters:
        - df (pd.DataFrame): Input frequency-transformed DataFrame.
        - cluster_columns (list): List of columns to use for clustering.
        - n_clusters (int): Number of clusters for final clustering.
        - k_range (tuple): Range of k values to determine optimal clustering (default: (2, 10)).
        
        Returns:
        - df_cluster (pd.DataFrame): DataFrame with added 'cluster' column.
        """
        df = self.apply_temporal_abstraction()
        df_cluster = df
        
        # Determine optimal k (inertia calculation)
        inertias = []
        for k in range(K_RANGE[0], K_RANGE[1]):
            subset = df_cluster[CLUSTER_COLUMNS]
            kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
            kmeans.fit(subset)
            inertias.append(kmeans.inertia_)
        
        # Apply clustering with the specified number of clusters
        kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=0)
        df_cluster["cluster"] = kmeans.fit_predict(df_cluster[CLUSTER_COLUMNS])
        logging.info("Process Clustering completed succesfully")
        return df_cluster

    def export_data(self):
        """Export processed data to pickle file."""
        try:
            df_clean = self.process_clustering()

            os.makedirs(os.path.dirname(self.feature_config.features_extracted_file_name), exist_ok=True)

            df_clean.to_pickle(self.feature_config.features_extracted_file_name)  # Export the df attribute
            logging.info(f"Data exported successfully to {self.feature_config.features_extracted_file_name}")

        except Exception as e:
            logging.error("Error in export_data method")
            raise MyException(e, sys)
        
    def export_features(self):
        try:
            logging.info("\nStarting features export process...")  
            logging.info(f"{'='*20} Features Extraction Started {'='*20}")          
            self.export_data()

            logging.info(f"{'='*20} Features Extraction Completed {'='*20}")
        except Exception as e:
            logging.error("\nERROR: Exception occurred in export_data method.")
            raise MyException(e, sys)




  