from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "Financial Document Processor"
    debug: bool = False
    log_level: str = "INFO"
    
    # OpenAI
    openai_api_key: str
    
    # Database
    database_url: str
    test_database_url: Optional[str] = None
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    
    # Document Processing
    max_file_size_mb: int = 50
    supported_formats: str = "pdf"
    ocr_language: str = "eng"
    
    class Config:
        env_file = ".env"


settings = Settings()
print("================================================================")
print(f"DEBUG: Loaded database_url = '{settings.database_url}'")
print(f"DEBUG: OpenAI key loaded = {'Yes' if settings.openai_api_key else 'No'}")
print("========================================================================================================")
