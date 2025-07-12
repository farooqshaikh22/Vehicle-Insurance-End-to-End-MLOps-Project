import os
import sys
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.insurance_data import InsuranceData
from sklearn.model_selection import train_test_split
from pandas import DataFrame


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise MyException(e, sys)
        
    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        
        try:
            logging.info("Exporting data from mongodb")
            my_data = InsuranceData()
            
            dataframe = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            logging.info(f"Shape of datafram: {dataframe.shape}")
            
            feature_store_filepath = self.data_ingestion_config.feature_store_filepath
            dir_name = os.path.dirname(feature_store_filepath)
            os.makedirs(dir_name, exist_ok=True)
            
            logging.info(f"Saving exported data to feature store filepath: {feature_store_filepath}")
            dataframe.to_csv(feature_store_filepath, index=False, header=True)
            
            return dataframe
        
        except Exception as e:
            raise MyException(e, sys)
        
    def split_data_as_train_test(self, dataframe:DataFrame)->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")
        
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed train-test-split on the dataframe")
            logging.info(f"Shape of train set: {train_set.shape}")
            logging.info(f"Shape of test set: {test_set.shape}")
            
            dir = os.path.dirname(self.data_ingestion_config.training_filepath)
            os.makedirs(dir, exist_ok=True)
            
            logging.info(f"Exporting data in train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_filepath,index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_filepath,index=False, header=True)
            logging.info(f"Exported data train and test file path.")

        except Exception as e:
            raise MyException(e, sys)
        
    def initiate_data_ingestion(self)->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("entered initiate_data_ingestion method of DataIngestion class")
        
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from mongodb")
            
            self.split_data_as_train_test(dataframe=dataframe)
            logging.info("Performed train test split on dataframe")
            
            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            
            data_ingestion_artifact = DataIngestionArtifact(training_filepath=self.data_ingestion_config.training_filepath, test_filepath=self.data_ingestion_config.test_filepath)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            
            return data_ingestion_artifact
 
        except Exception as e:
            raise MyException(e, sys)
        
