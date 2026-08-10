"""Optional audio-file transcription with no import-time model dependency."""

from __future__ import annotations

from pathlib import Path

from agentic_manipulation.errors import ConfigurationError


def transcribe_audio(path: Path, model_name: str = "small") -> str:
    audio_path = Path(path).expanduser()
    if not audio_path.is_file():
        raise ConfigurationError(f"audio file does not exist: {audio_path}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ConfigurationError(
            "voice input requires faster-whisper; install the project voice extra"
        ) from exc
    try:
        model = WhisperModel(model_name, device="auto", compute_type="int8_float16")
        segments, _ = model.transcribe(str(audio_path))
        text = "".join(segment.text for segment in segments).strip()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConfigurationError(f"audio transcription failed: {exc}") from exc
    if not text:
        raise ConfigurationError("audio transcription returned empty text")
    return text

