import os
import sys
import numpy as np
from typing import Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ClassificationMetricArtifact
from src.utils.main_utils import load_object, save_object, load_numpy_array, read_yaml_file
from src.entity.estimator import MyModel

class ModelTrainer:
    
    def __init__(self, data_transformation_artifact:DataTransformationArtifact, model_trainer_config:ModelTrainerConfig):
        """
        param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.model_config = read_yaml_file(filepath=self.model_trainer_config.model_config_filepath)
        
    def get_model_object_and_report(self, train:np.array, test:np.array)->Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("training RandomForestClassifier with specified parameters")
            
            ## splitting train and test data into features and target variable
            X_train, X_test, y_train, y_test = train[:, :-1], test[:, :-1], train[:, -1], test[:, -1]
            logging.info("train-test-split done")
            
            ## Initialize RandomForestClassifier with specified parameters
            params = self.model_config["parameters"]
            model = RandomForestClassifier(
                max_depth = params["max_depth"],
                n_estimators = params["n_estimators"],
                min_samples_split = params["min_samples_split"],
                min_samples_leaf = params["min_samples_leaf"],
                criterion = params["criterion"],
                random_state = params["random_state"]
            )
            
            ## fit the model
            logging.info("Starting model training")
            model.fit(X_train, y_train)
            logging.info("model training done")
            
            ## prediction and evealuation metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            ## creating metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
            
            return model, metric_artifact

        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            print("-----------------------------------------------------------------------")
            logging.info("starting model trainer component")
            
            ## load transformed train and test data
            logging.info("loading transformed train and test array")
            train_arr = load_numpy_array(filepath=self.data_transformation_artifact.transformed_train_filepath)
            test_arr = load_numpy_array(filepath=self.data_transformation_artifact.transformed_test_filepath)
            
            ## train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            ## load preprocessing object
            preprocessor_obj = load_object(filepath=self.data_transformation_artifact.transformed_object_filepath)
            logging.info("preprocessor object loaded")
            
            if accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) < self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")
            
            ## # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessor_obj, trained_model_object=trained_model)
            save_object(filepath=self.model_trainer_config.trained_model_filepath, obj=my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")
            
            ## model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_filepath=self.model_trainer_config.trained_model_filepath,
                metric_artifact=metric_artifact
            )
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys)