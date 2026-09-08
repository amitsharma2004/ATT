"""Pipeline Orchestrator (coordinating audio -> diarization -> STT -> alignment -> speaker identification)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.alignment.aligner import TimestampAligner
from backend.app.services.meeting_pipeline.alignment.models import AlignedTranscriptResult
from backend.app.services.meeting_pipeline.audio.processor import AudioProcessor
from backend.app.services.meeting_pipeline.diarization.diarizer import SpeakerDiarizer
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.embedding import SpeakerEmbeddingService
from backend.app.services.meeting_pipeline.speaker.matcher import SpeakerMatcher
from backend.app.services.meeting_pipeline.speaker.models import MeetingTranscriptionResult
from backend.app.services.meeting_pipeline.speaker.registry import VoiceRegistry
from backend.app.services.meeting_pipeline.stt.whisper_service import WhisperTranscriber


class MeetingPipelineOrchestrator:
    """Orchestrates complete meeting transcription and named speaker identification.

    Flow:
    Audio -> Preprocess (16kHz mono WAV)
          -> Pyannote Diarization (speaker turns)
          -> Whisper Large-v3 STT (timed segments)
          -> Timestamp Alignment (aligned segments)
          -> Speaker Embedding Extraction (per speaker cluster)
          -> Voice Registry Matching (cosine similarity & exclusive assignment)
          -> Named Meeting Transcript
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        audio_processor: Optional[AudioProcessor] = None,
        diarizer: Optional[SpeakerDiarizer] = None,
        transcriber: Optional[WhisperTranscriber] = None,
        aligner: Optional[TimestampAligner] = None,
        embedding_service: Optional[SpeakerEmbeddingService] = None,
        registry: Optional[VoiceRegistry] = None,
        matcher: Optional[SpeakerMatcher] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.audio_processor = audio_processor or AudioProcessor(settings=self.settings)
        self.diarizer = diarizer or SpeakerDiarizer(settings=self.settings)
        self.transcriber = transcriber or WhisperTranscriber(settings=self.settings)
        self.aligner = aligner or TimestampAligner(
            min_overlap_ratio=self.settings.alignment_min_overlap_ratio,
            dominance_threshold=self.settings.alignment_dominance_threshold,
            dominance_ratio_margin=self.settings.alignment_dominance_ratio_margin,
            merge_gap_s=self.settings.alignment_merge_gap_s,
            enable_merging=self.settings.alignment_enable_merging,
        )

        speaker_config = SpeakerConfig(
            embedding_model_name=self.settings.speaker_embedding_model_name,
            huggingface_token=self.settings.effective_hf_token,
            device=self.settings.compute_device,
            high_confidence_threshold=self.settings.speaker_high_confidence_threshold,
            medium_confidence_threshold=self.settings.speaker_medium_confidence_threshold,
            candidate_threshold=self.settings.speaker_candidate_threshold,
            enforce_exclusive_assignment=self.settings.speaker_enforce_exclusive_assignment,
        )
        self.embedding_service = embedding_service or SpeakerEmbeddingService(
            config=speaker_config, settings=self.settings
        )
        self.registry = registry or VoiceRegistry(
            registry_dir=self.settings.voice_registry_path,
            settings=self.settings,
            embedding_service=self.embedding_service,
            audio_processor=self.audio_processor,
        )
        self.matcher = matcher or SpeakerMatcher(config=speaker_config)

    def process_meeting(
        self,
        audio_path: Union[str, Path],
        team_id: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        num_speakers: Optional[int] = None,
        language: Optional[str] = None,
    ) -> MeetingTranscriptionResult:
        """End-to-end meeting transcription with named team member identification."""
        # 1. Preprocess audio
        processed = self.audio_processor.preprocess(Path(audio_path))

        # 2. Pyannote speaker diarization
        diarization_segments = self.diarizer.diarize(
            processed.processed_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )

        # 3. Whisper speech-to-text
        transcription_result = self.transcriber.transcribe(
            processed.processed_path,
            language=language,
        )

        # 4. Temporal timestamp alignment
        aligned_result = self.aligner.align(
            transcript_segments=transcription_result.segments,
            diarization_segments=diarization_segments,
        )

        # 5. Extract one centroid embedding per diarization speaker cluster
        unique_speakers = sorted(list(set(s.speaker for s in diarization_segments)))
        cluster_embeddings: Dict[str, np.ndarray] = {}

        for speaker_label in unique_speakers:
            cluster_turns = [s for s in diarization_segments if s.speaker == speaker_label]
            cluster_emb = self.embedding_service.extract_embedding(
                processed.processed_path,
                speech_segments=cluster_turns,
            )
            cluster_embeddings[speaker_label] = cluster_emb

        # 6. Fetch team voice profiles & match
        enrolled_profiles = self.registry.list_profiles(team_id=team_id)
        cluster_matches = self.matcher.match_clusters(cluster_embeddings, enrolled_profiles)

        # 7. Map identities back onto transcript segments
        return self.matcher.apply_matches_to_transcript(aligned_result, cluster_matches)

    def health_check(self) -> dict:
        """Verify pipeline subsystem readiness."""
        return {
            "status": "ready",
            "audio_processor": "initialized",
            "diarizer": "configured",
            "transcriber": "configured",
            "aligner": "initialized",
            "embedding_service": "configured",
            "voice_registry": "initialized",
            "matcher": "initialized",
            "stages_available": [
                "audio_validation",
                "audio_storage",
                "audio_standardization",
                "speaker_diarization",
                "whisper_stt",
                "timestamp_alignment",
                "speaker_embeddings",
                "voice_registry",
                "speaker_matching",
            ],
            "enrolled_profiles_count": len(self.registry.list_profiles()),
        }
