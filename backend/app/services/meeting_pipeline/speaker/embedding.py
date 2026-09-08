"""Speaker embedding extraction using pyannote embedding model."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.diarization.models import SpeakerSegment
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.exceptions import (
    AuthenticationError,
    EmbeddingError,
    ModelUnavailableError,
)

logger = logging.getLogger(__name__)


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2 normalize vector to unit length (norm == 1.0)."""
    norm = np.linalg.norm(vector)
    if norm == 0 or np.isnan(norm):
        return vector
    return vector / norm


def compute_centroid(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute L2-normalized centroid from multiple voice embedding vectors."""
    if not embeddings:
        raise EmbeddingError("Cannot compute centroid from empty list of embeddings")
    avg = np.mean(embeddings, axis=0)
    return l2_normalize(avg)


class SpeakerEmbeddingService:
    """Extracts speaker voice embeddings using pyannote's speaker encoder model.

    Features:
    - Lazy model loading & GPU device allocation
    - L2 normalization on all extracted embeddings
    - Centroid embedding computation across multiple speech samples
    - Speech interval cropping (extracts embeddings exclusively from active speech)
    """

    def __init__(
        self,
        config: Optional[SpeakerConfig] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()

        if config is None:
            self.config = SpeakerConfig(
                embedding_model_name=getattr(self.settings, "speaker_embedding_model_name", "pyannote/embedding"),
                huggingface_token=self.settings.effective_hf_token,
                device=self.settings.compute_device,
            )
        else:
            self.config = config

        self._inference: Any = None
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def initialize(self) -> None:
        """Load pyannote embedding model into memory."""
        if self._is_initialized and self._inference is not None:
            return

        try:
            import torch
            from pyannote.audio import Inference, Model
        except ImportError as exc:
            raise ModelUnavailableError(
                f"pyannote.audio or torch is not installed: {exc}"
            ) from exc

        token_str = (
            self.config.huggingface_token.get_secret_value()
            if self.config.huggingface_token
            else None
        )
        if not token_str:
            raise AuthenticationError(
                "Hugging Face token is missing for speaker embedding model. "
                "Please set HUGGINGFACE_TOKEN in .env."
            )

        try:
            logger.info("Loading speaker embedding model: %s", self.config.embedding_model_name)
            model = Model.from_pretrained(
                self.config.embedding_model_name,
                token=token_str,
            )
        except Exception as exc:
            msg = str(exc)
            if "gated" in msg.lower() or "401" in msg or "403" in msg or "unauthorized" in msg.lower():
                raise AuthenticationError(
                    f"Access not granted to gated model '{self.config.embedding_model_name}': {exc}"
                ) from exc
            raise ModelUnavailableError(
                f"Failed to load embedding model '{self.config.embedding_model_name}': {exc}"
            ) from exc

        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            inference = Inference(model, window="whole", device=torch.device(device_str))
            logger.info("Speaker embedding inference loaded on device: %s", device_str)
        except Exception as exc:
            logger.warning("Failed loading embedding inference on %s, falling back to CPU: %s", device_str, exc)
            inference = Inference(model, window="whole", device=torch.device("cpu"))

        self._inference = inference
        self._is_initialized = True

    def extract_embedding(
        self,
        audio_path: Union[str, Path],
        speech_segments: Optional[List[SpeakerSegment]] = None,
    ) -> np.ndarray:
        """Extract L2-normalized speaker embedding from audio.

        If speech_segments are provided, extracts embeddings across the active speech turns
        and averages them into a representative cluster vector.
        """
        path = Path(audio_path)
        if not path.exists():
            raise EmbeddingError(f"Audio file does not exist: {path}")

        if not self._is_initialized or self._inference is None:
            self.initialize()

        try:
            from pyannote.core import Segment

            # Get total audio duration to clamp crops safely (using built-in wave module)
            total_duration = None
            try:
                import wave
                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        total_duration = float(frames) / float(rate)
            except Exception:
                pass

            if speech_segments:
                segment_embeddings = []
                for seg in speech_segments:
                    start_s = max(0.0, float(seg.start))
                    end_s = float(seg.end)
                    if total_duration is not None:
                        end_s = min(end_s, total_duration)

                    dur = end_s - start_s
                    if dur < 0.5:  # Skip tiny fragments (< 500ms) for high-quality embedding
                        continue

                    crop = Segment(start_s, end_s)
                    emb = self._inference.crop(str(path), crop)
                    if hasattr(emb, "data"):
                        emb = emb.data
                    emb_arr = np.array(emb, dtype=np.float32).flatten()
                    if np.linalg.norm(emb_arr) > 0:
                        segment_embeddings.append(l2_normalize(emb_arr))

                if segment_embeddings:
                    return compute_centroid(segment_embeddings)

            # Whole audio file fallback (e.g. for enrollment clips)
            raw_emb = self._inference(str(path))
            if hasattr(raw_emb, "data"):
                raw_emb = raw_emb.data
            emb_arr = np.array(raw_emb, dtype=np.float32).flatten()
            return l2_normalize(emb_arr)

        except Exception as exc:
            raise EmbeddingError(f"Failed to extract embedding from {path.name}: {exc}") from exc
