from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite+aiosqlite:///./htr.db"
    database_url_sync: str = "sqlite:///./htr.db"


    trocr_model_id: str = "microsoft/trocr-base-handwritten"
    grammar_model_id: str = "vennify/t5-base-grammar-correction"
    device: str = "cpu"

    upload_dir: str = "./uploads"
    low_confidence_threshold: float = 0.55

    frontend_origin: str = "http://localhost:5173"


settings = Settings()
