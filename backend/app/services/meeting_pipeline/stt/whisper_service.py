"""Whisper Speech-to-Text service using Hugging Face Transformers pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


class WhisperTranscriber:
    """Encapsulates Whisper Large-v3 speech transcription using Hugging Face Transformers.

    Features:
    - Lazy loading and model reuse across requests
    - Automatic chunking (chunk_length_s + stride) for arbitrarily long recordings
    - Segment-level timestamp preservation for downstream alignment
    - Multilingual speech & automatic language detection
    - Memory-efficient 4-bit (NF4) / float16 inference optimized for 4GB VRAM constraint
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

        self._pipe: Any = None
        self._device: str = "cpu"
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """True if pipeline is loaded in memory."""
        return self._is_initialized

    @property
    def resolved_device(self) -> str:
        """Resolved device string ('cuda' or 'cpu')."""
        return self._device

    def initialize(self) -> None:
        """Load Whisper model and build Hugging Face pipeline."""
        if self._is_initialized and self._pipe is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ModelLoadingError(
                f"Required dependencies (torch/transformers) not installed: {exc}"
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
        logger.info(
            "Initializing WhisperTranscriber with model: %s on device: %s (quantization: %s)",
            self.config.model_name,
            self._device,
            self.config.quantization,
        )

        try:
            processor = AutoProcessor.from_pretrained(self.config.model_name)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Failed to load processor for Whisper model '{self.config.model_name}': {exc}"
            ) from exc

        try:
            model_kwargs: Dict[str, Any] = {
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
            }

            if self._device == "cuda":
                torch.cuda.empty_cache()

                if self.config.quantization == "4bit":
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                elif self.config.quantization == "8bit":
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = bnb_config
                    model_kwargs["device_map"] = "auto"
                else:
                    compute_dtype = torch.bfloat16 if self.config.torch_dtype == "bfloat16" else torch.float16
                    model_kwargs["torch_dtype"] = compute_dtype

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
                if self.config.quantization == "none":
                    model.to("cuda")
            else:
                model_kwargs["torch_dtype"] = torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
                model.to("cpu")

        except torch.cuda.OutOfMemoryError as exc:
            raise CUDAOutOfMemoryError(
                f"CUDA out of memory loading Whisper model '{self.config.model_name}'. "
                f"Hardware VRAM is insufficient for this precision: {exc}"
            ) from exc
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower():
                raise CUDAOutOfMemoryError(f"CUDA OOM loading Whisper: {exc}") from exc
            raise ModelLoadingError(
                f"Failed to load Whisper model '{self.config.model_name}': {exc}"
            ) from exc

        try:
            pipe_kwargs: Dict[str, Any] = {
                "task": "automatic-speech-recognition",
                "model": model,
                "tokenizer": processor.tokenizer,
                "feature_extractor": processor.feature_extractor,
                "chunk_length_s": self.config.chunk_length_s,
                "stride_length_s": (self.config.stride_length_s, self.config.stride_length_s),
            }
            if self.config.quantization == "none":
                pipe_kwargs["device"] = self._device

            self._pipe = pipeline(**pipe_kwargs)
            self._is_initialized = True
            logger.info("WhisperTranscriber initialized successfully")
        except Exception as exc:
            raise ModelLoadingError(
                f"Failed to build ASR pipeline for '{self.config.model_name}': {exc}"
            ) from exc

    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe a standardized 16kHz mono audio recording.

        Args:
            audio_path: Path to canonical audio file (from Step 2).
            language: Optional language hint; if None, Whisper automatically detects language.

        Returns:
            TranscriptionResult with segment-level timestamps.
        """
        path = Path(audio_path)
        if not path.exists():
            raise InvalidAudioPathError(f"Audio file does not exist: {path}")

        if not self._is_initialized or self._pipe is None:
            self.initialize()

        generate_kwargs: Dict[str, Any] = {
            "task": "transcribe",
        }
        lang = language or self.config.language
        if lang:
            generate_kwargs["language"] = lang

        try:
            raw_output = self._pipe(
                str(path),
                return_timestamps=True,
                generate_kwargs=generate_kwargs,
                batch_size=self.config.batch_size,
            )
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda oom" in msg.lower():
                raise CUDAOutOfMemoryError(f"CUDA Out of Memory during transcription: {exc}") from exc
            raise TranscriptionError(f"Whisper transcription failed on {path.name}: {exc}") from exc

        return self._process_pipe_output(raw_output, path)

    def _process_pipe_output(self, raw_output: Any, path: Path) -> TranscriptionResult:
        """Parse raw pipeline dictionary into validated TranscriptionResult and TranscriptionSegments."""
        raw_chunks = raw_output.get("chunks", []) if isinstance(raw_output, dict) else []
        segments: List[TranscriptionSegment] = []

        for chunk in raw_chunks:
            text = chunk.get("text", "").strip()
            timestamp = chunk.get("timestamp")
            if not text:
                continue

            if isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start, end = timestamp
                start_val = float(start) if start is not None else 0.0
                end_val = float(end) if end is not None else (start_val + 0.1)
                # Ensure end > start
                if end_val <= start_val:
                    end_val = start_val + 0.1

                segments.append(
                    TranscriptionSegment(
                        start=round(start_val, 3),
                        end=round(end_val, 3),
                        text=text,
                    )
                )

        # In case chunks was empty but full text was generated (e.g. short audio without chunks)
        if not segments and isinstance(raw_output, dict) and raw_output.get("text", "").strip():
            full_txt = raw_output["text"].strip()
            segments.append(
                TranscriptionSegment(
                    start=0.0,
                    end=1.0,
                    text=full_txt,
                )
            )

        # Sort segments chronologically
        segments.sort(key=lambda s: s.start)

        detected_lang = None
        if isinstance(raw_output, dict):
            detected_lang = raw_output.get("language")

        audio_duration = segments[-1].end if segments else 0.0

        return TranscriptionResult(
            segments=segments,
            language=detected_lang,
            duration_seconds=round(audio_duration, 3),
            model_name=self.config.model_name,
            device_used=self._device,
            metadata={
                "chunk_length_s": self.config.chunk_length_s,
                "stride_length_s": self.config.stride_length_s,
                "quantization": self.config.quantization,
            },
        )
