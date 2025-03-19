import os
import sys
import wandb
from dotenv import load_dotenv

from src.exception import MyException
from src.logger import logging

from src.constants import *
from src.entity.configEntity import *
from src.data.data_ingestion import IngestData
from src.data.data_processing import MakeDataset
from src.data.removeOutlier import RemoveOutlier
from src.features.buildFeatures import FeaturesExtraction
from src.models.modelTrainer import ModelTrainer

load_dotenv()

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_processing_config = MakeDatasetConfig()
        self.outlier_removing_config = RemoveOutlierConfig()
        self.features_config = FeaturesExtractionConfig()
        self.model_config = ModelTrainerConfig()
        
        # Initialize WandB
        wandb.init(project="model-training-pipeline", name="train-pipeline", config={
            "data_source": os.getenv("DATA_SOURCE"),
            "model_type": os.getenv("MODEL_TYPE"),
        })

    def start_data_ingestion(self):
        try:
            logging.info("Starting data ingestion.")
            wandb.log({"stage": "Data Ingestion"})
            
            data_ingestion = IngestData(data_ingestion_config=self.data_ingestion_config, use_wandb=True)
            data_ingestion_artifact = data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            
            wandb.log({"data_ingestion_status": "Completed"})
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys)

    def start_data_processing(self):
        try:
            logging.info("Starting data processing.")
            wandb.log({"stage": "Data Processing"})
            
            data_processor = MakeDataset(make_dataset_config=self.data_processing_config, use_wandb=True)
            processed_data = data_processor.process_and_save()
            
            
            wandb.log({"data_processing_status": "Completed"})
            return processed_data
        except Exception as e:
            raise MyException(e, sys)

    def start_removing_outliers(self):
        try:
            logging.info("Removing outliers.")
            wandb.log({"stage": "Outlier Removal"})
            
            outlier = RemoveOutlier(outlier_removing_config=self.outlier_removing_config, use_wandb=True)
            outlier_df = outlier.remove_outliers(method=METHOD_CHAUVENET)
            outlier.export_data()
            
            wandb.log({"outlier_removal_status": "Completed"})
            return outlier_df
        except Exception as e:
            raise MyException(e, sys)

    def start_features_engg(self):
        try:
            logging.info("Extracting features.")
            wandb.log({"stage": "Feature Engineering"})
            
            feature = FeaturesExtraction(features_extraction_config=self.features_config)
            export_features = feature.export_features()
            
            wandb.log({"feature_engineering_status": "Completed"})
            return export_features
        except Exception as e:
            raise MyException(e, sys)

    def start_model_training(self):
        try:
            logging.info("Starting model training.")
            wandb.log({"stage": "Model Training"})
            
            trainer = ModelTrainer(model_trainer_config=self.model_config)
            trainer.load_data()
            trainer.split_data_as_train_test()
            trainer.prepare_feature_set()
            trainer.grid_search_model_selection()
            
            wandb.log({"model_training_status": "Completed"})
            return
        except Exception as e:
            raise MyException(e, sys)

    def run_pipeline(self) -> None:
        try:
            logging.info("Running the training pipeline.")
            wandb.log({"pipeline_status": "Started"})
            
            self.start_data_ingestion()
            self.start_data_processing()
            self.start_removing_outliers()
            self.start_features_engg()
            self.start_model_training()
            
            wandb.log({"pipeline_status": "Completed"})
            logging.info("Training pipeline completed successfully.")
            return 
        except Exception as e:
            raise MyException(e, sys)
        finally:
            wandb.finish()
