import os
import sys
import wandb
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

from src.exception import MyException
from src.logger import logging
from src.constants import *
from src.entity.configEntity import FeaturesExtractionConfig
from src.utils.dataTransformation import LowPassFilter, PrincipalComponentAnalysis
from src.utils.temporalAbstraction import NumericalAbstraction
from src.utils.frequencyAbstraction import FourierTransformation
from src.data.removeOutlier import RemoveOutlier


class FeaturesExtraction:
    """
    A class to handle the complete feature extraction process for sensor data.
    
    Includes:
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
        Initialize Feature Extraction with logging, error handling, and WandB.

        Args:
            features_extraction_config (FeaturesExtractionConfig): Configuration object containing file paths and parameters.
        """
        try:
            self.feature_config = features_extraction_config
            
            # Initialize WandB
            wandb.init(project="sensor_feature_extraction", name="feature_pipeline", config={
                "dataset": self.feature_config.root_dir,
                "low_pass_cutoff": CUTTOFF,
                "sampling_frequency": FS,
                "num_pca_components": 3,
                "window_size": WS,
                "num_clusters": N_CLUSTERS
            })

            logging.info("WandB initialized successfully.")

        except Exception as e:
            logging.error("Error in initializing Features Extraction pipeline")
            raise MyException(e, sys)

    def load_data(self):
        """Load sensor data from a pickle file."""
        try:
            file_name = self.feature_config.root_dir
            df = pd.read_pickle(file_name) 

            logging.info("Data loaded successfully")
            wandb.log({"data_shape": df.shape})  # Log dataset shape
            return df

        except Exception as e:
            logging.error("Error in loading data")
            raise MyException(e, sys)

    def handle_missing_values(self):
        """Handle missing values using interpolation."""
        try:
            df = self.load_data()
            predictor_columns = PREDICTOR_COLUMNS

            for col in predictor_columns:
                df[col] = df[col].interpolate()

            logging.info("Missing values handled successfully")
            return df
        except Exception as e:
            logging.error("Error in handling missing values")
            raise MyException(e, sys)

    def perform_pca(self):
        """Perform PCA for dimensionality reduction."""
        try:
            df = self.handle_missing_values()
            pca = PrincipalComponentAnalysis()

            # Log explained variance
            explained_variance = pca.determine_pc_explained_variance(df, PREDICTOR_COLUMNS)
            wandb.log({"pca_explained_variance": explained_variance})

            # Apply PCA
            df_pca = pca.apply_pca(df, PREDICTOR_COLUMNS, 3)

            logging.info("PCA applied successfully")
            return df_pca
        except Exception as e:
            logging.error("Error in PCA processing")
            raise MyException(e, sys)

    def process_lowpass_filter(self):
        """Apply a low-pass Butterworth filter."""
        try:
            df = self.perform_pca()
            lowpass = LowPassFilter()

            for col in PREDICTOR_COLUMNS:
                df = lowpass.low_pass_filter(df, col, FS, CUTTOFF, order=5)
                df[col] = df[col + "_lowpass"]
                del df[col + "_lowpass"]

            logging.info("Low-pass filter applied successfully")
            return df
        except Exception as e:
            logging.error("Error in low-pass filter")
            raise MyException(e, sys)

    def apply_temporal_abstraction(self):
        """Apply temporal abstraction on sensor data."""
        try:
            df = self.process_lowpass_filter()
            num_abs = NumericalAbstraction()

            predictor_columns = PREDICTOR_COLUMNS + ["acc_r", "gyr_r"]
            for col in predictor_columns:
                df = num_abs.abstract_numerical(df, [col], WS, "mean")
                df = num_abs.abstract_numerical(df, [col], WS, "std")

            logging.info("Temporal abstraction applied successfully")
            return df
        except Exception as e:
            logging.error("Error in temporal abstraction")
            raise MyException(e, sys)

    def process_clustering(self):
        """Apply KMeans clustering."""
        try:
            df = self.apply_temporal_abstraction()
            inertias = []

            for k in range(K_RANGE[0], K_RANGE[1]):
                kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
                kmeans.fit(df[CLUSTER_COLUMNS])
                inertias.append(kmeans.inertia_)

            # Log clustering inertia values
            wandb.log({"clustering_inertia": inertias})

            # Apply final clustering
            kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=0)
            df["cluster"] = kmeans.fit_predict(df[CLUSTER_COLUMNS])

            logging.info("Clustering applied successfully")
            return df
        except Exception as e:
            logging.error("Error in clustering process")
            raise MyException(e, sys)

    def export_data(self):
        """Export processed data to pickle file and log to WandB."""
        try:
            df_clean = self.process_clustering()
            os.makedirs(os.path.dirname(self.feature_config.features_extracted_file_name), exist_ok=True)

            df_clean.to_pickle(self.feature_config.features_extracted_file_name)
            wandb.log({"exported_data_shape": df_clean.shape})  # Log final data shape
            
            # Save processed data to WandB as an artifact
            artifact = wandb.Artifact('processed_sensor_data', type='dataset')
            artifact.add_file(self.feature_config.features_extracted_file_name)
            wandb.log_artifact(artifact)

            logging.info(f"Data exported successfully to {self.feature_config.features_extracted_file_name}")
        except Exception as e:
            logging.error("Error in export_data method")
            raise MyException(e, sys)

    def export_features(self):
        """Main feature extraction pipeline runner."""
        try:
            logging.info("\nStarting features export process...")  
            logging.info(f"{'='*20} Features Extraction Started {'='*20}")          
            self.export_data()

            logging.info(f"{'='*20} Features Extraction Completed {'='*20}")
            wandb.finish()  # Ensure WandB run is properly closed
        except Exception as e:
            logging.error("\nERROR: Exception occurred in export_data method.")
            raise MyException(e, sys)


  