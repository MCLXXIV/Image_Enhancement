"""Фабрика стадий: возвращает словарь со всеми загруженными Enhancer'ами."""

from __future__ import annotations

from enhancer.models.base import Enhancer
from enhancer.models.clahe import ClaheEnhancer
from enhancer.models.denoise import NlmDenoiseEnhancer
from enhancer.models.gamma import GammaEnhancer
from enhancer.models.unsharp import UnsharpEnhancer
from enhancer.models.whitebalance import GrayWorldEnhancer
from enhancer.quality.router import Stage


def build_default_stages() -> dict[Stage, Enhancer]:
    return {
        Stage.GAMMA: GammaEnhancer(),
        Stage.CLAHE: ClaheEnhancer(),
        Stage.WHITE_BALANCE: GrayWorldEnhancer(),
        Stage.UNSHARP: UnsharpEnhancer(),
        Stage.DENOISE: NlmDenoiseEnhancer(),
    }
