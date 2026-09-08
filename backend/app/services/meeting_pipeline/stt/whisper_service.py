"""Whisper Speech-to-Text service using faster-whisper (CTranslate2 backend)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.stt.config import STTConfig
from backend.app.services.meeting_pipeline.stt.exceptions import (
    CUDANotAvailableError,
    CUDAOutOfMemoryError,
    InvalidAudioPathError,
    ModelLoadingError,
    ModelUnavailableError,
    STTError,
    TranscriptionError,
)
from backend.app.services.meeting_pipeline.stt.models import (
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

# Maps OpenAI Whisper Hub identifiers to faster-whisper (CTranslate2) model size strings.
_FASTER_WHISPER_MODEL_MAP = {
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large-v2": "large-v2",
    "openai/whisper-large-v1": "large-v1",
    "openai/whisper-large": "large",
    "openai/whisper-medium": "medium",
    "openai/whisper-medium.en": "medium.en",
    "openai/whisper-small": "small",
    "openai/whisper-small.en": "small.en",
    "openai/whisper-base": "base",
    "openai/whisper-base.en": "base.en",
    "openai/whisper-tiny": "tiny",
    "openai/whisper-tiny.en": "tiny.en",
}


class WhisperTranscriber:
    """Encapsulates Whisper Large-v3 speech transcription using faster-whisper (CTranslate2).

    Features:
    - Lazy loading and model reuse across requests
    - Native long-form transcription with internal VAD-based segmentation
      (avoids the HF transformers pipeline's chunk-stitching timestamp bugs)
    - Segment-level timestamp preservation for downstream alignment
    - Multilingual speech & automatic language detection
    - int8 quantization for efficient CPU inference; float16 on GPU
    """

    def __init__(
        self,
        config: Optional[STTConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()

        if config is None:
            self.config = STTConfig(
                model_name=self.settings.whisper_model_name,
                device=self.settings.compute_device,
                torch_dtype=self.settings.whisper_torch_dtype,
                quantization=self.settings.whisper_quantization,
                chunk_length_s=self.settings.whisper_chunk_length_s,
                stride_length_s=self.settings.whisper_stride_length_s,
            )
        else:
            self.config = config

        self._model: Any = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """True if the model is loaded in memory."""
        return self._is_initialized

    @property
    def resolved_device(self) -> str:
        """Resolved device string ('cuda' or 'cpu')."""
        return self._device

    @staticmethod
    def _resolve_model_size(model_name: str) -> str:
        """Map an OpenAI Whisper Hub id to its faster-whisper model size string.

        Falls back to the raw model_name (faster-whisper also accepts CTranslate2
        repo ids like 'Systran/faster-whisper-large-v3' or local paths directly).
        """
        return _FASTER_WHISPER_MODEL_MAP.get(model_name, model_name)

    def initialize(self) -> None:
        """Load the faster-whisper model into memory."""
        if self._is_initialized and self._model is not None:
            return

        try:
            import torch
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ModelLoadingError(
                f"Required dependency faster-whisper (or torch) is not installed: {exc}"
            ) from exc

        # Determine target device
        requested = self.config.device
        if requested == "cuda":
            if not torch.cuda.is_available():
                raise CUDANotAvailableError(
                    "CUDA device was explicitly requested but torch.cuda.is_available() is False"
                )
            target_device = "cuda"
        elif requested == "cpu":
            target_device = "cpu"
        else:  # auto
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = target_device

        # Resolve compute type: int8 quantization for lean memory footprint,
        # full precision only when quantization is explicitly disabled.
        if target_device == "cuda":
            compute_type = "float16" if self.config.quantization == "none" else "int8_float16"
        else:
            compute_type = "float32" if self.config.quantization == "none" else "int8"
        self._compute_type = compute_type

        model_size = self._resolve_model_size(self.config.model_name)

        logger.info(
            "Initializing WhisperTranscriber (faster-whisper) with model: %s on device: %s (compute_type: %s)",
            model_size,
            self._device,
            compute_type,
        )

        try:
            self._model = WhisperModel(
                model_size,
                device=self._device,
                compute_type=compute_type,
            )
            self._is_initialized = True
            logger.info("WhisperTranscriber initialized successfully")
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                raise CUDAOutOfMemoryError(f"CUDA OOM loading Whisper: {exc}") from exc
            if "gated" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
                raise ModelUnavailableError(
                    f"Failed to access faster-whisper model '{model_size}': {exc}"
                ) from exc
            raise ModelLoadingError(
                f"Failed to load faster-whisper model '{model_size}': {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe a standardized 16kHz mono audio recording.

        Args:
            audio_path: Path to canonical audio file (from Step 2).
            language: Optional language hint; if None, faster-whisper automatically detects language.

        Returns:
            TranscriptionResult with segment-level timestamps.
        """
        path = Path(audio_path)
        if not path.exists():
            raise InvalidAudioPathError(f"Audio file does not exist: {path}")

        if not self._is_initialized or self._model is None:
            self.initialize()

        lang = language or self.config.language

        try:
            segments_iter, info = self._model.transcribe(
                str(path),
                language=lang,
                task="transcribe",
                vad_filter=True,
                beam_size=5,
            )
            raw_segments = list(segments_iter)
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda oom" in msg.lower():
                raise CUDAOutOfMemoryError(f"CUDA Out of Memory during transcription: {exc}") from exc
            raise TranscriptionError(f"Whisper transcription failed on {path.name}: {exc}") from exc

        return self._process_segments(raw_segments, info)

    def _process_segments(self, raw_segments: List[Any], info: Any) -> TranscriptionResult:
        """Convert faster-whisper Segment objects into validated TranscriptionSegments."""
        segments: List[TranscriptionSegment] = []

        for seg in raw_segments:
            text = (getattr(seg, "text", "") or "").strip()
            if not text:
                continue

            start_val = float(seg.start)
            end_val = float(seg.end)
            if end_val <= start_val:
                end_val = start_val + 0.1

            segments.append(
                TranscriptionSegment(
                    start=round(start_val, 3),
                    end=round(end_val, 3),
                    text=text,
                )
            )

        # faster-whisper yields segments chronologically already; sort defensively.
        segments.sort(key=lambda s: s.start)

        detected_lang = getattr(info, "language", None)
        audio_duration = getattr(info, "duration", None)
        if audio_duration is None:
            audio_duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            duration_seconds=round(float(audio_duration), 3),
            model_name=self.config.model_name,
            device_used=self._device,
            metadata={
                "engine": "faster-whisper",
                "compute_type": self._compute_type,
            },
        )
