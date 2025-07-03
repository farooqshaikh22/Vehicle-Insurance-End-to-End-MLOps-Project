import os
import sys
import pymongo
import certifi
from src.logger import logging
from src.exception import MyException
from src.constants import DB_NAME, CONNECTION_URL_KEY
from dotenv import load_dotenv

load_dotenv()

# Load the certificate authority file to avoid timeout errors when connecting to MongoDB
ca = certifi.where()

class MongoDBclient:
    """
    MongoDBClient is responsible for establishing a connection to the MongoDB database.

    Attributes:
    ----------
    client : MongoClient
        A shared MongoClient instance for the class.
    database : Database
        The specific database instance that MongoDBClient connects to.

    Methods:
    -------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the given database name.
    
    """
    client = None  # Shared MongoClient instance across all MongoDBClient instances
    
    def __init__(self, database_name:str = DB_NAME):
        """
        Initializes a connection to the MongoDB database. If no existing connection is found, it establishes a new one.

        Parameters:
        ----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is set by DATABASE_NAME constant.

        Raises:
        ------
        MyException
            If there is an issue connecting to MongoDB or if the environment variable for the MongoDB URL is not set.
        
        """
        
        # Check if a MongoDB client connection has already been established; if not, create a new one
        
        try:
            if MongoDBclient.client is None:
                mongo_db_url = os.getenv(CONNECTION_URL_KEY)
                
                if mongo_db_url is None:
                    raise Exception(f"environment variable '{CONNECTION_URL_KEY}' is not set")
                
                MongoDBclient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            # Use the shared MongoClient for this instance
            self.client = MongoDBclient.client
            self.database_name = database_name
            self.database = self.client[database_name]
            logging.info("MongoDB connection successful")
            
        except Exception as e:
            raise MyException(e, sys)
            