"""Central configuration using Pydantic Settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline and application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General application settings
    app_name: str = "STT-Speaker-Pipeline"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True

    # Storage paths
    storage_dir: Path = Path("storage")
    upload_dir_name: str = "raw"
    processed_dir_name: str = "processed"

    # Audio preprocessing specification
    target_sample_rate: int = 16000
    target_channels: int = 1
    max_audio_size_mb: int = 500
    allowed_audio_extensions: List[str] = Field(
        default=["wav", "mp3", "m4a", "flac", "ogg", "aac", "wma"]
    )
    audio_highpass_cutoff: float = 75.0
    audio_enable_denoise: bool = True
    audio_target_loudness_dbfs: float = -20.0
    audio_low_loudness_dbfs: float = -30.0
    audio_noise_floor_threshold_dbfs: float = -35.0
    audio_clip_threshold: float = 0.99
    audio_silence_threshold_dbfs: float = -50.0


    # Diarization settings (pyannote)
    huggingface_token: Optional[SecretStr] = Field(default=None, alias="HUGGINGFACE_TOKEN")
    hf_token: Optional[SecretStr] = Field(default=None, alias="HF_TOKEN")
    diarization_model_name: str = "pyannote/speaker-diarization-3.1"
    diarization_merge_distance_sec: float = 0.25
    diarization_device: Literal["cuda", "cpu", "auto"] = Field(default="cpu", alias="DIARIZATION_DEVICE")

    # STT Whisper settings
    whisper_model_name: str = "openai/whisper-large-v3"
    whisper_chunk_length_s: float = 30.0
    whisper_stride_length_s: float = 5.0
    whisper_torch_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    whisper_quantization: Literal["4bit", "8bit", "none"] = "4bit"

    # Alignment settings
    alignment_min_overlap_ratio: float = 0.25
    alignment_dominance_threshold: float = 0.60
    alignment_dominance_ratio_margin: float = 0.15
    alignment_merge_gap_s: float = 0.50
    alignment_enable_merging: bool = True

    # Speaker Embedding & Matching settings
    speaker_embedding_model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM"
    speaker_high_confidence_threshold: float = 0.85
    speaker_medium_confidence_threshold: float = 0.75
    speaker_candidate_threshold: float = 0.60
    speaker_enforce_exclusive_assignment: bool = True

    # Sarvam Batch STT Settings
    sarvam_api_key: Optional[SecretStr] = Field(default=None, alias="SARVAM_API_KEY")
    sarvam_model: str = Field(default="saaras:v4", alias="SARVAM_MODEL")
    sarvam_language_code: str = Field(default="en-IN", alias="SARVAM_LANGUAGE_CODE")
    sarvam_mode: str = Field(default="transcribe", alias="SARVAM_MODE")
    sarvam_with_diarization: bool = Field(default=True, alias="SARVAM_WITH_DIARIZATION")
    sarvam_poll_interval_seconds: int = Field(default=5, alias="SARVAM_POLL_INTERVAL_SECONDS")
    sarvam_max_wait_seconds: int = Field(default=600, alias="SARVAM_MAX_WAIT_SECONDS")

    # Compute device
    compute_device: Literal["cuda", "cpu", "auto"] = "auto"

    @property
    def effective_hf_token(self) -> Optional[SecretStr]:
        """Resolve token from either HUGGINGFACE_TOKEN or HF_TOKEN."""
        return self.huggingface_token or self.hf_token

    @property
    def raw_storage_path(self) -> Path:
        return self.storage_dir / self.upload_dir_name

    @property
    def processed_storage_path(self) -> Path:
        return self.storage_dir / self.processed_dir_name

    @property
    def voice_registry_path(self) -> Path:
        return self.storage_dir / "voice_registry"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
