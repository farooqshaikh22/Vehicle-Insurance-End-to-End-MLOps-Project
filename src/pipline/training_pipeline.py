import sys
from src.logger import logging
from src.exception import MyException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import (DataIngestionConfig,
                                      DataValidationConfig,
                                      DataTransformationConfig,
                                      ModelTrainerConfig
                                      )

from src.entity.artifact_entity import (DataIngestionArtifact,
                                        DataValidationArtifact,
                                        DataTransformationArtifact,
                                        ModelTrainerArtifact
                                        )


class TrainPipeline: 
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        
    def start_data_ingestion(self)->DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component.
        """
        
        try:
            logging.info("Entered the start_data_ingestion method of TrainPipeline class")
            logging.info("Getting the data from MongoDB")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train set and test set from MongoDB")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            
            return data_ingestion_artifact
        
        except Exception as e:
            raise MyException(e, sys)
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data validation component.
        """
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, 
                                             data_validation_config=self.data_validation_config)
            
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info("Performed the data validation operation")
            logging.info("Exited the start_data_validation method of TrainPipeline class")
            
            return data_validation_artifact
                
        except Exception as e:
            raise MyException(e, sys)
        
    def start_data_transformation(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_artifact:DataValidationArtifact)->DataValidationArtifact:
        """
        This method of TrainPipeline class is responsible for starting data transformation component.
        """
        try:
            logging.info("Entered the start_data_validation method of TrainPipeline class")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact,data_transformation_config=self.data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info("Performed the data transformation")
            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            
            return data_transformation_artifact
        
        except Exception as e:
            raise MyException(e, sys)
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact):
        """
        This method of TrainPipeline class is responsible for starting model trainer component.
        """
        try:
            logging.info("Entered the start_model_trainer method of TrainPipeline class")
            model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,model_trainer_config=self.model_trainer_config)
            
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info("Performed model training")
            logging.info("Exited the start_model_trainer method of TrainPipeline class")
            
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys)
        
    def run_pipeline(self)->None:
        """
        This method of train pipeline is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact=data_ingestion_artifact,data_validation_artifact=data_validation_artifact)
            
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        
        except Exception as e:
            raise MyException(e, sys)
        
    
