import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from src.constants import SCHEMA_FILE_PATH, TARGET_COLUMN, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_data, read_yaml_file, save_numpy_array, save_object


class DataTransformation:
    
    def __init__(self, data_ingestion_artifact:DataIngestionArtifact, data_validation_artifact:DataValidationArtifact, data_transformation_config:DataTransformationConfig):
        
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self.schema_config = read_yaml_file(filepath=SCHEMA_FILE_PATH)
             
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data_transformer_object(self)-> Pipeline:
        """
        Creates and returns a data preprocessing pipeline using ColumnTransformer.

    The pipeline applies:
        - StandardScaler to numeric features
        - MinMaxScaler to specified columns
        - Passes through remaining columns unchanged

    Returns:
        Pipeline: A scikit-learn Pipeline object containing the preprocessing steps.
        
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        
        try:
            ## initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
            
            ## load schema configuration
            num_features = self.schema_config["num_features"]
            mm_columns = self.schema_config["mm_columns"]
            logging.info("Cols loaded from schema.")
            
            ## creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder="passthrough"
            )
            
            ## wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
            logging.info("final pipeline ready")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            
            return final_pipeline   
 
        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise (e, sys)
        
    def map_gender_column(self, df):
        """ Map Gender column to 0 for Female and 1 for Male. """
        logging.info("Mapping 'Gender' column to binary values")
        df["Gender"] = df["Gender"].map({"Female":0, "Male":1}).astype(int)
        return df
    
    def create_dummy_columns(self, df):
        """Create dummy variables for categorical features."""
        logging.info("Creating dummy variables for categorical features")
        df = pd.get_dummies(df, drop_first=True)
        return df
    
    def rename_columns(self, df):
        """ Rename specific columns and ensure integer types for dummy columns. """
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
        
        for col in ['Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years', 'Vehicle_Damage_Yes']:
            if col in df.columns:
                df[col] = df[col].astype('int')
                
        return df
    
    def drop_id_column(self, df):
        """ drop the 'id' column if it exist """
        logging.info("dropping 'id' column")
        drop_col = self.schema_config["drop_columns"]
        if drop_col in df.columns:
            df = df.drop(columns=drop_col, axis=1)
        
        return df
    
    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data transformation started")
            if not self.data_validation_artifact.validation_status:
                logging.error(self.data_validation_artifact.message)
            
            ## load train-test data
            logging.info("loading train and test data")
            train_df = read_data(file_path=self.data_ingestion_artifact.training_filepath)
            test_df = read_data(file_path=self.data_ingestion_artifact.test_filepath)
            logging.info("train and test data loaded")
            
            ## defining input-feature columns and output variable column
            logging.info("defining input and output feature columns for train_df")
            X_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_train_df = train_df[TARGET_COLUMN]
            
            logging.info("defining input and output feature columns for test_df")
            X_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            y_test_df = test_df[TARGET_COLUMN]
            
            ## Apply custom transformations in specified sequence
            logging.info("Apply custom transformations on training input features in specified sequence")
            X_train_df = self.drop_id_column(X_train_df)
            X_train_df = self.map_gender_column(X_train_df)
            X_train_df = self.create_dummy_columns(X_train_df)
            X_train_df = self.rename_columns(X_train_df)
            
            logging.info("Apply custom transformations on testing input features in specified sequence")
            X_test_df= self.drop_id_column(X_test_df)
            X_test_df = self.map_gender_column(X_test_df)
            X_test_df = self.create_dummy_columns(X_test_df)
            X_test_df= self.rename_columns(X_test_df)
            logging.info("custom transformation applied to training and test data")
            
            logging.info("starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")
            
            logging.info("Initializing transformation for training data")
            X_train_array = preprocessor.fit_transform(X_train_df)
            logging.info("Initializing transformation for test data")
            X_test_array = preprocessor.transform(X_test_df)
            logging.info("Transformation done to train-test data")
            
            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            X_train_final, y_train_final = smt.fit_resample(X_train_array, y_train_df)
            X_test_final, y_test_final = smt.fit_resample(X_test_array, y_test_df)
            logging.info("SMOTEENN applied to train-test df.")
            
            train_arr = np.c_[X_train_final, np.array(y_train_final)]
            test_arr = np.c_[X_test_final, np.array(y_test_final)]
            logging.info("feature-target concatenation done for train-test df.")
            
            logging.info("Saving transformation object and transformed files.")
            save_object(filepath=self.data_transformation_config.transformed_object_filepath, obj=preprocessor)
            save_numpy_array(filepath=self.data_transformation_config.transformed_train_filepath, array=train_arr)
            save_numpy_array(filepath=self.data_transformation_config.transformed_test_filepath, array=test_arr)
            logging.info("Successfully  saved transformation object and transformed files.")
            
            logging.info("Data transformation completed successfully")
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_filepath=self.data_transformation_config.transformed_object_filepath,
                transformed_train_filepath=self.data_transformation_config.transformed_train_filepath,
                transformed_test_filepath=self.data_transformation_config.transformed_test_filepath
            )
            
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys)
    