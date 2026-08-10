"""Environment-driven configuration with no implicit downloads."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from .errors import ConfigurationError


@dataclass(frozen=True)
class RuntimeConfig:
    ollama_url: str = "http://127.0.0.1:11434"
    qwen_model: str = "qwen3-vl:2b"
    graspnet_checkpoint: Path | None = None
    max_retries: int = 2
    mode: str = "mock"

    def __post_init__(self) -> None:
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ConfigurationError("ollama_url must use http:// or https://")
        if not self.qwen_model.strip():
            raise ConfigurationError("qwen_model must not be empty")
        if self.max_retries < 0:
            raise ConfigurationError("AGENTIC_MAX_RETRIES must be non-negative")
        if self.mode not in {"mock", "real"}:
            raise ConfigurationError("AGENTIC_MODE must be 'mock' or 'real'")

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        raw_retries = os.getenv("AGENTIC_MAX_RETRIES", "2")
        try:
            max_retries = int(raw_retries)
        except ValueError as exc:
            raise ConfigurationError(
                "AGENTIC_MAX_RETRIES must be a non-negative integer"
            ) from exc

        checkpoint = os.getenv("AGENTIC_GRASPNET_CHECKPOINT")
        return cls(
            ollama_url=os.getenv(
                "AGENTIC_OLLAMA_URL", "http://127.0.0.1:11434"
            ).rstrip("/"),
            qwen_model=os.getenv("AGENTIC_QWEN_MODEL", "qwen3-vl:2b"),
            graspnet_checkpoint=Path(checkpoint) if checkpoint else None,
            max_retries=max_retries,
            mode=os.getenv("AGENTIC_MODE", "mock").lower(),
        )
