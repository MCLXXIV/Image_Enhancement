from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENHANCER_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    s3_endpoint_url: str = Field(default="http://minio:9000", alias="S3_ENDPOINT_URL")

    # --- Real-SAFMN++ (SR + restoration на мелких/низкоразрешённых фото) ---
    safmn_x2_weights_path: str | None = Field(default=None, alias="SAFMN_X2_WEIGHTS_PATH")
    safmn_x4_weights_path: str | None = Field(default=None, alias="SAFMN_X4_WEIGHTS_PATH")
    safmn_device: str | None = Field(default=None, alias="SAFMN_DEVICE")
    safmn_tile: int = Field(default=256, alias="SAFMN_TILE")
    safmn_fp16: bool = Field(default=False, alias="SAFMN_FP16")
    safmn_dim: int = Field(default=128, alias="SAFMN_DIM")
    safmn_n_blocks: int = Field(default=16, alias="SAFMN_N_BLOCKS")
    safmn_ffn_scale: float = Field(default=2.0, alias="SAFMN_FFN_SCALE")
    # Доля SR-выхода в результате: 1.0 только модель, <1.0 подмешивает бикубик (меньше пластика).
    safmn_strength: float = Field(default=0.8, alias="SAFMN_STRENGTH")

    # --- Retinexformer (low-light: экспозиция / шум / цвет) ---
    lowlight_weights_path: str | None = Field(default=None, alias="LOWLIGHT_WEIGHTS_PATH")
    lowlight_device: str | None = Field(default=None, alias="LOWLIGHT_DEVICE")
    # Конфиг сети под чекпоинт (LOL_v2_real: n_feat=40, stage=1, num_blocks=1,2,2).
    lowlight_n_feat: int = Field(default=40, alias="LOWLIGHT_N_FEAT")
    lowlight_stage: int = Field(default=1, alias="LOWLIGHT_STAGE")
    lowlight_num_blocks: str = Field(default="1,2,2", alias="LOWLIGHT_NUM_BLOCKS")
    # Доля выхода модели в результате: 1.0 только модель, <1.0 подмешивает оригинал.
    lowlight_strength: float = Field(default=0.8, alias="LOWLIGHT_STRENGTH")

    # --- IAT (exposure) ---
    exposure_weights_path: str | None = Field(default=None, alias="EXPOSURE_WEIGHTS_PATH")
    exposure_device: str | None = Field(default=None, alias="EXPOSURE_DEVICE")

    # --- SCUNet (restoration scale=1: шум / JPEG / лёгкий блюр на больших фото) ---
    restore_weights_path: str | None = Field(default=None, alias="RESTORE_WEIGHTS_PATH")
    restore_device: str | None = Field(default=None, alias="RESTORE_DEVICE")
    restore_tile: int = Field(default=256, alias="RESTORE_TILE")
    restore_dim: int = Field(default=64, alias="RESTORE_DIM")
    # Конфиг блоков SCUNet под real-чекпоинт scunet_color_real_psnr.pth (CSV).
    restore_config: str = Field(default="4,4,4,4,4,4,4", alias="RESTORE_CONFIG")

    # --- Роутер: пороги размеров ---
    low_res_max_side: int = Field(default=1280, alias="LOW_RES_MAX_SIDE")
    sr_input_max_side: int = Field(default=2048, alias="SR_INPUT_MAX_SIDE")
    sr_target_long_side: int = Field(default=1920, alias="SR_TARGET_LONG_SIDE")

    # --- IQA-verify (really improved?) ---
    iqa_device: str | None = Field(default=None, alias="IQA_DEVICE")
    iqa_gate_enabled: bool = Field(default=True, alias="IQA_GATE_ENABLED")


settings = Settings()
