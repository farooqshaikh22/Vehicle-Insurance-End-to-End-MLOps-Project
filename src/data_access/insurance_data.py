import pandas as pd
import numpy as np
import sys
from typing import Optional
from src.logger import logging
from src.exception import MyException
from src.configuration.mongo_db_connection import MongoDBclient
from src.constants import DB_NAME,COLLECTION_NAME

class InsuranceData:
    """
    A class to export mongodb records as pandas dataframe
    """
    def __init__(self)->None:
        """ initialize mongodb client connection"""
        
        try:
            self.mongo_client = MongoDBclient(database_name=DB_NAME)
        except Exception as e:
            raise MyException(e, sys)
        
    def export_collection_as_dataframe(self, collection_name:str=COLLECTION_NAME, database_name:Optional[str]=None)->pd.DataFrame:
        
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
                
            else:
                collection = self.mongo_client[database_name][collection_name]
                
            ## convert collection data to dataframe and preprocess
            logging.info("Fetching data from MongoDB")
            df = pd.DataFrame(list(collection.find()))
            logging.info(f"data fetched successfully from mongodb with length: {len(df)}")
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)
                
            df.replace({"na":np.nan}, inplace=True)
            return df
        
        except Exception as e:
            raise MyException(e, sys)
                
        
            