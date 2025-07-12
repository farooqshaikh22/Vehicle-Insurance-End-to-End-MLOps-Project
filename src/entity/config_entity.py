import os
from dataclasses import dataclass
from src.constants import *
from datetime import datetime

TIMESTAMP:str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = PIPELINE_NAME
    artifact_dir:str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp:str = TIMESTAMP
    
training_pipeline_config:TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_filepath:str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_filepath:str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    test_filepath:str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio:float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    
    
@dataclass
class DataValidationConfig:
    data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    data_validation_report_file_path:str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)
    

@dataclass
class DataTransformationConfig:
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_filepath:str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, DATA_TRANSFORMATION_TRANSFORMED_TRAIN_FILE_NAME)
    transformed_test_filepath:str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, DATA_TRANSFORMATION_TRANSFORMED_TEST_FILE_NAME) 
    transformed_object_filepath:str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCSSING_OBJECT_FILE_NAME)  
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_filepath:str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy:float = MODEL_TRAINER_EXPECTED_SCORE
    model_config_filepath:str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH