import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

## constants for log configuration
LOG_DIR = "logs"
LOG_DIR_PATH = os.path.join(os.getcwd(), LOG_DIR)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Construct log file path
os.makedirs(LOG_DIR_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR_PATH, LOG_FILE)

MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

def configure_logger():
    """  Configures logging with a rotating file handler and a console handler. """
    
    ## create a custom logger
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    
    ## file handler with rotation
    file_handler = RotatingFileHandler(filename=LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setLevel("DEBUG")
    
    ## console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel("DEBUG")
    
    ## formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    ## Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
# Configure the logger
configure_logger()