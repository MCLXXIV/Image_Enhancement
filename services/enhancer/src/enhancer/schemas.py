from pydantic import BaseModel


class EnhanceParams(BaseModel):
    """Опциональные отладочные оверрайды. По умолчанию пайплайн полностью автоматический."""

    force: bool | None = None  # применить даже если фото «хорошее» + отключить IQA-fallback
    force_lowlight: bool | None = None  # принудительно прогнать Zero-DCE++
    force_restore: bool | None = None  # принудительно прогнать SCUNet
    force_safmn: bool | None = None  # принудительно прогнать Real-SAFMN++


class HealthResponse(BaseModel):
    status: str
    version: str
