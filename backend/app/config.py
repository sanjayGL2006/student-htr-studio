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

    trocr_model_id: str = str(BASE_DIR / "models" / "trocr-custom") if (BASE_DIR / "models" / "trocr-custom").exists() else "microsoft/trocr-base-handwritten"
    grammar_model_id: str = "vennify/t5-base-grammar-correction"
    device: str = "cpu"

    upload_dir: str = str(BASE_DIR / "uploads")
    low_confidence_threshold: float = 0.55

    frontend_origin: str = "http://localhost:5173"


settings = Settings()
