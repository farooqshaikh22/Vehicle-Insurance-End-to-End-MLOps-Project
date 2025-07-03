import os
import sys
import numpy as np
import yaml
import dill
import pandas as pd
from pandas import DataFrame
from src.logger import logging
from src.exception import MyException


def read_data(file_path: str) -> DataFrame:
    """
    Reads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        DataFrame: Loaded data as a pandas DataFrame.

    Raises:
        MyException: If reading the file fails.
    """
    logging.info(f"Attempting to read CSV file from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df

    except FileNotFoundError:
        logging.error(f"CSV file not found: {file_path}")
        raise MyException(f"File not found: {file_path}", sys)

    except pd.errors.EmptyDataError:
        logging.error(f"CSV file is empty: {file_path}")
        raise MyException("Empty CSV file", sys)

    except pd.errors.ParserError as e:
        logging.error(f"Parser error while reading CSV: {e}")
        raise MyException(f"CSV parsing error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while reading CSV: {e}")
        raise MyException(e, sys) from e



def read_yaml_file(filepath:str)->dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed content of the YAML file.

    Raises:
        MyException: If the file is not found, contains YAML errors, or any other exception occurs.
    """
    logging.info(f"Reading yaml file from path: {filepath}")
    
    try:
        with open(filepath, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info("YAML file read successfully.")
        
            return content
    
    except FileNotFoundError:
        logging.error(f"YAML file not found: {filepath}")
        raise MyException(f"File not found: {filepath}", sys)

    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error in file {filepath}: {e}")
        raise MyException(f"YAML Error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while reading YAML file: {e}")
        raise MyException(e, sys) from e
    


def write_yaml_file(filepath:str, content:object, replace:bool=False)->None:
    """
    Writes Python object content to a YAML file.

    Args:
        file_path (str): Path to save the YAML file.
        content (object): Python object (usually dict) to write into YAML.
        replace (bool, optional): Whether to replace the file if it already exists. Defaults to False.

    Raises:
        MyException: If any error occurs during file writing.
    """
    try:
        if replace and os.path.exists(filepath):
            os.remove(filepath)
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logging.info(f"Writing YAML content to: {filepath}")
        
        with open(filepath, "w") as file:
            yaml.dump(content, file)
            logging.info("YAML fiile written successfully")
    
    except PermissionError:
        logging.error(f"Permission denied while writing to: {filepath}")
        raise MyException("Permission denied", sys)

    except yaml.YAMLError as e:
        logging.error(f"Error dumping YAML content: {e}")
        raise MyException(f"YAML dumping error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while writing YAML file: {e}")
        raise MyException(e, sys) from e
    
    

def save_object(filepath:str, obj:object)->None:
    """
    Serializes and saves a Python object to the given file path using `dill`.

    Args:
        file_path (str): Path to save the serialized object.
        obj (object): Python object to serialize and save.

    Raises:
        MyException: If saving fails due to permission issues or unexpected errors.
    """
    logging.info(f"Attempting to save object to: {filepath}")
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as file_obj:
            dill.dump(obj, file_obj)
            logging.info("object saved successfully")
    
    except PermissionError:
        logging.error(f"Permission denied while saving to: {filepath}")
        raise MyException(f"Permission denied: {filepath}", sys)

    except dill.PicklingError as e:
        logging.error(f"Dill pickling error: {e}")
        raise MyException(f"Pickling error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while saving object: {e}")
        raise MyException(e, sys) from e
    
    

def load_object(filepath:str)->object:
    """
    Loads a serialized object (e.g., model, transformer) from the specified file path using dill.

    Args:
        file_path (str): Path to the file containing the serialized object.

    Returns:
        object: The deserialized Python object.

    Raises:
        MyException: If the file is not found, corrupted, or any other error occurs during loading.
    """  
    logging.info(f"Attempting to load object from: {filepath}")
    try:
        with open(filepath, "rb") as file_obj:
            obj = dill.load(file_obj)
            logging.info("object loaded successfully")
            return obj
    
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise MyException(f"File not found: {filepath}", sys)

    except dill.UnpicklingError as e:
        logging.error(f"Dill unpickling error while loading object: {e}")
        raise MyException(f"Dill unpickling error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while loading object: {e}")
        raise MyException(e, sys) from e  
    
    
    
def save_numpy_array(filepath:str, array:np.array)->None:
    """
    Saves a NumPy array to a binary `.npy` file using NumPy's save method.

    Args:
        file_path (str): Path to save the NumPy array file.
        array (np.array): NumPy array to save.

    Raises:
        MyException: If an error occurs while saving the array.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        logging.info(f"Saving NumPy array to: {filepath}")
        
        with open(filepath, "wb") as file_obj:
            np.save(file_obj, array)
            logging.info("numpy array saved successfully")
    
    except PermissionError:
        logging.error(f"Permission denied when writing to: {filepath}")
        raise MyException(f"Permission denied: {filepath}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while saving NumPy array: {e}")
        raise MyException(e, sys) from e
    
    

def load_numpy_array(filepath:str)->np.array:
    """
    Loads a NumPy array from a `.npy` binary file.

    Args:
        file_path (str): Path to the NumPy file.

    Returns:
        np.ndarray: Loaded NumPy array.

    Raises:
        MyException: If the file is not found or loading fails.
    """
    logging.info(f"Attempting to load NumPy array from: {filepath}")
    
    try:
        with open(filepath, "rb") as file_obj:
            array = np.load(file_obj)
            logging.info("numpy array loaded successfully")
            return array
    
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise MyException(f"File not found: {filepath}", sys)

    except dill.UnpicklingError as e:
        logging.error(f"Dill unpickling error while loading object: {e}")
        raise MyException(f"Dill unpickling error: {e}", sys)

    except Exception as e:
        logging.error(f"Unexpected error while loading object: {e}")
        raise MyException(e, sys) from e