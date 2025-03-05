from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    raw_data_file_path:str

@dataclass
class DataProcessingArtifact:
    # data_processing_status: bool
    processed_file_path: str
    is_processed: bool
    message: str
