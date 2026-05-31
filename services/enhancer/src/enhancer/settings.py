from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENHANCER_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")

    safmn_weights_path: str | None = Field(default=None, alias="SAFMN_WEIGHTS_PATH")
    safmn_scale: int = Field(default=4, alias="SAFMN_SCALE")
    safmn_device: str | None = Field(default=None, alias="SAFMN_DEVICE")
    safmn_tile: int = Field(default=256, alias="SAFMN_TILE")
    safmn_fp16: bool = Field(default=False, alias="SAFMN_FP16")


settings = Settings()
