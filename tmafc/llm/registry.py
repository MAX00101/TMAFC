"""LLM backend factory."""
from __future__ import annotations

from typing import Callable, Dict

from tmafc.llm.base import BaseLLM
from tmafc.utils.logging import get_logger

logger = get_logger(__name__)

_REGISTRY: Dict[str, Callable[..., BaseLLM]] = {}


def register_backend(name: str):
    def deco(fn: Callable[..., BaseLLM]):
        _REGISTRY[name.lower()] = fn
        return fn
    return deco


@register_backend("mock")
def _build_mock(**cfg) -> BaseLLM:
    from tmafc.llm.mock_backend import MockLLM
    return MockLLM(
        agreement=float(cfg.get("agreement", 0.85)),
        seed=int(cfg.get("seed", 42)),
    )


@register_backend("openai")
def _build_openai(**cfg) -> BaseLLM:
    from tmafc.llm.openai_backend import OpenAIChatLLM
    return OpenAIChatLLM(
        model=cfg.get("model", "gpt-4"),
        api_key=cfg.get("api_key"),
        base_url=cfg.get("base_url"),
        timeout=float(cfg.get("timeout", 60.0)),
    )


def from_config(cfg: dict) -> BaseLLM:
    cfg = dict(cfg or {})
    name = str(cfg.pop("backend", "mock")).lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown LLM backend '{name}'; registered: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**cfg)


make_llm = from_config
