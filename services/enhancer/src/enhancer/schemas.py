from pydantic import BaseModel, Field


class EnhanceParams(BaseModel):
    force: bool | None = None
    gamma: float | None = Field(default=None, ge=0.5, le=2.0)
    clahe_clip: float | None = Field(default=None, ge=1.0, le=6.0)
    sharp_amount: float | None = Field(default=None, ge=0.0, le=2.0)
    denoise_strength: float | None = Field(default=None, ge=0.0, le=1.0)
    use_safmn: bool | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
