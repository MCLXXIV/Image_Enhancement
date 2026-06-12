from pydantic import BaseModel


class EnhanceParams(BaseModel):
    """Опциональные отладочные оверрайды. По умолчанию пайплайн полностью автоматический."""

    force: bool | None = None
    only: bool | None = None
    force_lowlight: bool | None = None
    force_exposure: bool | None = None
    force_restore: bool | None = None
    force_safmn: bool | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
