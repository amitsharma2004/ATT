"""Speaker Diarizer service wrapping pyannote.audio."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.diarization.config import DiarizationConfig
from backend.app.services.meeting_pipeline.diarization.exceptions import (
    DiarizationAuthenticationError,
    DiarizationError,
    DiarizationInferenceError,
    InvalidAudioPathError,
    ModelUnavailableError,
    PyannoteNotInstalledError,
)
from backend.app.services.meeting_pipeline.diarization.models import (
    DiarizationResult,
    SpeakerSegment,
)

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Encapsulates Pyannote speaker diarization with lazy model loading,

    deterministic speaker label normalization, and adjacent turn merging.
    Compatible with both Pyannote 3.x (returning Annotation) and Pyannote 4.x
    (returning DiarizeOutput with .speaker_diarization Annotation).
    """

    def __init__(
        self,
        config: Optional[DiarizationConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()

        if config is None:
            self.config = DiarizationConfig(
                model_name=self.settings.diarization_model_name,
                huggingface_token=self.settings.effective_hf_token,
                device=self.settings.diarization_device,
                merge_distance_sec=self.settings.diarization_merge_distance_sec,
            )
        else:
            self.config = config

        self._pipeline: Any = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return True if model is loaded into memory."""
        return self._is_initialized

    def initialize(self) -> None:
        """Explicitly load the pyannote pipeline into memory."""
        if self._is_initialized and self._pipeline is not None:
            return

        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise PyannoteNotInstalledError(
                f"pyannote.audio or torch is not installed: {exc}. Please install pyannote.audio."
            ) from exc

        token_str = (
            self.config.huggingface_token.get_secret_value()
            if self.config.huggingface_token
            else None
        )
        if not token_str:
            raise DiarizationAuthenticationError(
                "Hugging Face token is missing. Please set HUGGINGFACE_TOKEN or HF_TOKEN in environment or .env."
            )

        try:
            logger.info("=" * 60)
            logger.info("⏳ [DIARIZATION INIT] Loading Pyannote pipeline: %s", self.config.model_name)
            logger.info("ℹ️  NOTE: If this is the FIRST run, downloading Pyannote models (~1.5 GB).")
            logger.info("   The system is NOT crashed/frozen. Please wait...")
            logger.info("=" * 60)
            t0 = time.time()
            pipeline = Pipeline.from_pretrained(
                self.config.model_name,
                token=token_str,
            )
            logger.info("✅ [DIARIZATION READY] Pyannote pipeline loaded in %.2fs!", time.time() - t0)
        except Exception as exc:
            msg = str(exc)
            if "gated" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
                raise DiarizationAuthenticationError(
                    f"Authentication failed or access not granted to gated model '{self.config.model_name}'. "
                    f"Visit Hugging Face to accept terms: {exc}"
                ) from exc
            raise ModelUnavailableError(
                f"Failed to load Pyannote model '{self.config.model_name}': {exc}"
            ) from exc

        if pipeline is None:
            raise ModelUnavailableError(
                f"Pyannote Pipeline.from_pretrained returned None for model '{self.config.model_name}'"
            )

        # Device placement
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            pipeline.to(torch.device(device_str))
            logger.info("Pyannote pipeline moved to device: %s", device_str)
        except Exception as exc:
            logger.warning("Failed to move pipeline to %s, falling back to CPU: %s", device_str, exc)
            pipeline.to(torch.device("cpu"))

        self._pipeline = pipeline
        self._is_initialized = True

    def diarize(
        self,
        audio_path: Union[str, Path],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """Run diarization on the standardized audio file.

        Args:
            audio_path: Path to canonical 16kHz mono WAV file.
            min_speakers: Optional lower bound on speaker count.
            max_speakers: Optional upper bound on speaker count.
            num_speakers: Optional exact speaker count constraint.

        Returns:
            List of normalized, chronologically sorted, merged SpeakerSegment objects.
        """
        path = Path(audio_path)
        if not path.exists():
            raise InvalidAudioPathError(f"Audio file does not exist: {path}")

        if not self._is_initialized or self._pipeline is None:
            self.initialize()

        params: Dict[str, Any] = {}
        if num_speakers is not None:
            params["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                params["min_speakers"] = min_speakers
            if max_speakers is not None:
                params["max_speakers"] = max_speakers

        try:
            diarization_output = self._pipeline(str(path), **params)
        except Exception as exc:
            msg = str(exc)
            if "out of memory" in msg.lower() or "cuda" in msg.lower():
                logger.warning(
                    "CUDA OOM detected during Pyannote diarization (%s). Automatically falling back to CPU...",
                    msg[:120]
                )
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self._pipeline.to(torch.device("cpu"))
                    diarization_output = self._pipeline(str(path), **params)
                except Exception as cpu_exc:
                    raise DiarizationInferenceError(
                        f"Pyannote diarization inference failed on CPU fallback for {path.name}: {cpu_exc}"
                    ) from cpu_exc
            else:
                raise DiarizationInferenceError(
                    f"Pyannote diarization inference failed on {path.name}: {exc}"
                ) from exc

        return self._process_annotation(diarization_output)

    def _process_annotation(self, output: Any) -> List[SpeakerSegment]:
        """Extract turns, normalize speaker labels deterministically, and merge adjacent same-speaker turns."""
        raw_turns: List[Tuple[float, float, str]] = []

        # In pyannote 4.x, output is DiarizeOutput with speaker_diarization attribute
        # In pyannote 3.x, output is directly the Annotation
        annotation = getattr(output, "speaker_diarization", output)

        # Pyannote Annotation itertracks(yield_label=True)
        itertracks = getattr(annotation, "itertracks", None)
        if callable(itertracks):
            for segment, _, speaker_label in itertracks(yield_label=True):
                raw_turns.append((float(segment.start), float(segment.end), str(speaker_label)))
        elif hasattr(annotation, "tracks"):
            # Fallback for mock/dict representations
            for track in annotation.tracks:
                raw_turns.append((float(track.start), float(track.end), str(track.speaker)))

        # Sort raw segments chronologically by start time
        raw_turns.sort(key=lambda x: x[0])

        # Step 7: Normalize pyannote's raw labels into deterministic SPEAKER_00, SPEAKER_01...
        label_mapping: Dict[str, str] = {}
        next_speaker_idx = 0

        mapped_segments: List[Tuple[float, float, str]] = []
        for start, end, raw_label in raw_turns:
            if raw_label not in label_mapping:
                label_mapping[raw_label] = f"SPEAKER_{next_speaker_idx:02d}"
                next_speaker_idx += 1
            mapped_segments.append((start, end, label_mapping[raw_label]))

        # Step 8: Merge adjacent segments for the same speaker with no meaningful gap
        merged = self._merge_adjacent_segments(
            mapped_segments,
            max_gap=self.config.merge_distance_sec,
        )

        return [
            SpeakerSegment(
                speaker=spk,
                start=round(s, 3),
                end=round(e, 3),
                duration=round(e - s, 3),
            )
            for s, e, spk in merged
            if round(e - s, 3) > 0.0
        ]

    def _merge_adjacent_segments(
        self,
        segments: List[Tuple[float, float, str]],
        max_gap: float,
    ) -> List[Tuple[float, float, str]]:
        """Merge consecutive segments belonging to the exact same speaker if gap <= max_gap.

        Never merges across another speaker.
        """
        if not segments:
            return []

        merged: List[Tuple[float, float, str]] = []
        curr_start, curr_end, curr_spk = segments[0]

        for nxt_start, nxt_end, nxt_spk in segments[1:]:
            if nxt_spk == curr_spk and (nxt_start - curr_end) <= max_gap and nxt_start >= curr_start:
                # Merge into current segment
                curr_end = max(curr_end, nxt_end)
            else:
                merged.append((curr_start, curr_end, curr_spk))
                curr_start, curr_end, curr_spk = nxt_start, nxt_end, nxt_spk

        merged.append((curr_start, curr_end, curr_spk))
        return merged
