"""Импорт vendored архитектуры SAFMN из services/enhancer/."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

_REPO = Path(__file__).resolve().parents[3]
_ENHANCER_SRC = _REPO / "services" / "enhancer" / "src"
if str(_ENHANCER_SRC) not in sys.path:
    sys.path.insert(0, str(_ENHANCER_SRC))

from enhancer.models._safmn_arch import SAFMN as _SAFMN  # noqa: E402

SAFMN = cast(type, _SAFMN)
