from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_filepath:str
    test_filepath:str
    
@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    data_validation_report_file_path:str