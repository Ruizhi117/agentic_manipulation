"""User input adapters for the agentic manipulation runtime."""

from .text import run_text_loop
from .voice import transcribe_audio

__all__ = ["run_text_loop", "transcribe_audio"]

