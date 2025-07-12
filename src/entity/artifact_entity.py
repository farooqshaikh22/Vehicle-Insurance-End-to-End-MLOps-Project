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
    
@dataclass
class DataTransformationArtifact:
    transformed_object_filepath:str
    transformed_train_filepath:str
    transformed_test_filepath:str
    
@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    
@dataclass
class ModelTrainerArtifact:
    trained_model_filepath:str
    metric_artifact:ClassificationMetricArtifact
    
    