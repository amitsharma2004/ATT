"""Configuration options for Speech-to-Text stage."""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class STTConfig(BaseModel):
    """Configuration specific to the Whisper STT stage."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str = "openai/whisper-large-v3"
    device: Literal["cuda", "cpu", "auto"] = "auto"
    torch_dtype: Literal["float16", "float32", "bfloat16"] = "float16"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    chunk_length_s: float = Field(
        default=30.0,
        gt=0.0,
        description="Chunk length in seconds for long audio streaming",
    )
    stride_length_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Stride overlap length in seconds for chunk stitching",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Inference batch size (keep 1 for low VRAM 4GB GPUs)",
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code hint if known, or None for automatic language detection",
    )
