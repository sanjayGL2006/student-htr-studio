from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = f"sqlite+aiosqlite:///{BASE_DIR / 'htr.db'}"
    database_url_sync: str = f"sqlite:///{BASE_DIR / 'htr.db'}"
    
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    trocr_model_id: str = str(BASE_DIR / "models" / "trocr-custom") if (BASE_DIR / "models" / "trocr-custom").exists() else "microsoft/trocr-base-handwritten"
    grammar_model_id: str = "vennify/t5-base-grammar-correction"
    device: str = "cpu"

    upload_dir: str = str(BASE_DIR / "uploads")
    max_upload_size_mb: int = 20
    low_confidence_threshold: float = 0.55

    # Vision parameters
    dilation_kernel_width: int = 35
    dilation_kernel_height: int = 5
    min_line_height: int = 15
    max_line_height: int = 200
    min_line_width: int = 50
    line_padding: int = 12
    deskew_threshold: float = -45.0

    # NLP parameters
    enable_grammar_correction: bool = True
    enable_symspell: bool = True
    protected_tokens: str = "Python,JavaScript,React,PostgreSQL,BCA,SPVM3"

    # Authentication
    secret_key: str = "super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7 # 7 days

    frontend_origin: str = "http://localhost:5173"


settings = Settings()
