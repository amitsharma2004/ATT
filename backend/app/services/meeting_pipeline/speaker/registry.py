"""Voice Registry storing and managing team member voiceprints."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.audio.processor import AudioProcessor
from backend.app.services.meeting_pipeline.speaker.embedding import (
    SpeakerEmbeddingService,
    compute_centroid,
    l2_normalize,
)
from backend.app.services.meeting_pipeline.speaker.exceptions import (
    ProfileAlreadyExistsError,
    ProfileNotFoundError,
    RegistryError,
)
from backend.app.services.meeting_pipeline.speaker.models import VoiceProfile

logger = logging.getLogger(__name__)


class VoiceRegistry:
    """Manages team voiceprints with multi-sample enrollment and binary vector persistence.

    Directory Layout:
    storage/
      voice_registry/
        arjun/
          profile.json     (Metadata)
          embedding.bin    (Float32 raw binary vector)
          sample_1.wav     (Standardized audio clips)
    """

    def __init__(
        self,
        registry_dir: Optional[Path] = None,
        settings: Optional[Settings] = None,
        embedding_service: Optional[SpeakerEmbeddingService] = None,
        audio_processor: Optional[AudioProcessor] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry_dir = registry_dir or (self.settings.storage_dir / "voice_registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_service = embedding_service or SpeakerEmbeddingService(settings=self.settings)
        self.audio_processor = audio_processor or AudioProcessor(settings=self.settings)
        self._cache: Dict[str, VoiceProfile] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load all registered voice profiles from disk into memory cache."""
        self._cache.clear()
        if not self.registry_dir.exists():
            return

        for user_folder in self.registry_dir.iterdir():
            if not user_folder.is_dir():
                continue
            meta_path = user_folder / "profile.json"
            bin_path = user_folder / "embedding.bin"

            if meta_path.exists() and bin_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    with open(bin_path, "rb") as f:
                        bin_data = f.read()

                    profile = VoiceProfile.from_binary_bytes(
                        bin_data,
                        user_id=meta["user_id"],
                        name=meta["name"],
                        team_id=meta.get("team_id", "default"),
                        embedding_model=meta["embedding_model"],
                        samples_count=meta.get("samples_count", 1),
                        created_at=meta.get("created_at"),
                        updated_at=meta.get("updated_at"),
                    )
                    self._cache[profile.user_id] = profile
                except Exception as exc:
                    logger.warning("Failed loading profile from %s: %s", user_folder, exc)

    def enroll_team_member(
        self,
        user_id: str,
        name: str,
        sample_audio_paths: List[Union[str, Path]],
        team_id: str = "default",
        overwrite: bool = False,
    ) -> VoiceProfile:
        """Enroll a team member by extracting and centroid-averaging embeddings across 3-5 audio clips."""
        if not sample_audio_paths:
            raise RegistryError(f"Cannot enroll user '{user_id}' without audio samples")

        if user_id in self._cache and not overwrite:
            raise ProfileAlreadyExistsError(
                f"User '{user_id}' is already enrolled in the voice registry. Use overwrite=True to update."
            )

        user_dir = self.registry_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        sample_embeddings: List[np.ndarray] = []

        # Process each audio sample through Step 2 standardization and extract embedding
        for idx, s_path in enumerate(sample_audio_paths, start=1):
            path = Path(s_path)
            if not path.exists():
                raise RegistryError(f"Enrollment audio sample does not exist: {path}")

            # Standardize clip to 16kHz mono WAV
            proc_audio = self.audio_processor.preprocess(path, job_id=f"enroll_{user_id}_{idx}")
            dest_sample = user_dir / f"sample_{idx}.wav"
            shutil.copy2(proc_audio.processed_path, dest_sample)

            # Extract normalized embedding
            emb = self.embedding_service.extract_embedding(dest_sample)
            sample_embeddings.append(emb)

        # Compute robust centroid embedding
        centroid_emb = compute_centroid(sample_embeddings)
        dimension = len(centroid_emb)
        model_name = self.embedding_service.config.embedding_model_name

        profile = VoiceProfile(
            user_id=user_id,
            name=name,
            team_id=team_id,
            embedding=centroid_emb.tolist(),
            embedding_model=model_name,
            embedding_dimension=dimension,
            samples_count=len(sample_embeddings),
        )

        # Persist binary vector (~768B) and JSON metadata
        with open(user_dir / "embedding.bin", "wb") as f:
            f.write(profile.to_binary_bytes())

        meta = {
            "user_id": profile.user_id,
            "name": profile.name,
            "team_id": profile.team_id,
            "embedding_model": profile.embedding_model,
            "embedding_dimension": profile.embedding_dimension,
            "samples_count": profile.samples_count,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
        with open(user_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._cache[user_id] = profile
        logger.info("Successfully enrolled user '%s' (%s) with %d samples", name, user_id, len(sample_embeddings))
        return profile

    def register_profile_directly(self, profile: VoiceProfile, overwrite: bool = False) -> None:
        """Register a pre-constructed VoiceProfile directly (useful for testing and seed data)."""
        if profile.user_id in self._cache and not overwrite:
            raise ProfileAlreadyExistsError(f"Profile '{profile.user_id}' already exists")

        user_dir = self.registry_dir / profile.user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        with open(user_dir / "embedding.bin", "wb") as f:
            f.write(profile.to_binary_bytes())

        meta = {
            "user_id": profile.user_id,
            "name": profile.name,
            "team_id": profile.team_id,
            "embedding_model": profile.embedding_model,
            "embedding_dimension": profile.embedding_dimension,
            "samples_count": profile.samples_count,
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat(),
        }
        with open(user_dir / "profile.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._cache[profile.user_id] = profile

    def get_profile(self, user_id: str) -> VoiceProfile:
        """Retrieve voice profile by user_id."""
        if user_id not in self._cache:
            raise ProfileNotFoundError(f"Voice profile not found for user: '{user_id}'")
        return self._cache[user_id]

    def list_profiles(self, team_id: Optional[str] = None) -> List[VoiceProfile]:
        """List all enrolled voice profiles, optionally filtered by team_id."""
        if team_id:
            return [p for p in self._cache.values() if p.team_id == team_id]
        return list(self._cache.values())

    def delete_profile(self, user_id: str) -> None:
        """Remove voice profile and stored samples from disk and cache."""
        if user_id not in self._cache:
            raise ProfileNotFoundError(f"Cannot delete non-existent profile: '{user_id}'")

        user_dir = self.registry_dir / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)

        del self._cache[user_id]
        logger.info("Deleted voice profile for user: '%s'", user_id)
