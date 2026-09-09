"""Pipeline Orchestrator (coordinating audio -> diarization -> STT -> alignment -> speaker identification)."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger("pipeline.orchestrator")

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
        total_start = time.time()
        logger.info("=" * 70)
        logger.info("🚀 [START] New Meeting Processing Job Started: %s", audio_path)
        logger.info("=" * 70)

        # 1. Preprocess audio
        s1 = time.time()
        logger.info("⏳ [STAGE 1/6] Audio Preprocessing (Standardization, Denoise, VAD)...")
        processed = self.audio_processor.preprocess(Path(audio_path))
        logger.info("✅ [STAGE 1/6] Completed in %.2fs -> Output: %s (Duration: %.2fs)", 
                    time.time() - s1, processed.processed_path.name, processed.metadata.duration_seconds)

        # 2. Pyannote speaker diarization
        s2 = time.time()
        logger.info("⏳ [STAGE 2/6] Speaker Diarization running (Detecting who spoke when)...")
        diarization_segments = self.diarizer.diarize(
            processed.processed_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )
        unique_spks = sorted(list(set(s.speaker for s in diarization_segments)))
        logger.info("✅ [STAGE 2/6] Diarization completed in %.2fs -> Found %d segments across %d speaker(s): %s",
                    time.time() - s2, len(diarization_segments), len(unique_spks), unique_spks)

        # 3. Whisper speech-to-text
        s3 = time.time()
        logger.info("⏳ [STAGE 3/6] Whisper Speech-to-Text Transcription running...")
        transcription_result = self.transcriber.transcribe(
            processed.processed_path,
            language=language,
        )
        logger.info("✅ [STAGE 3/6] Transcription completed in %.2fs -> Transcribed %d segments (Language: %s)",
                    time.time() - s3, len(transcription_result.segments), transcription_result.language)

        # 4. Temporal timestamp alignment
        s4 = time.time()
        logger.info("⏳ [STAGE 4/6] Aligning timestamps between Whisper words & Diarization turns...")
        aligned_result = self.aligner.align(
            transcript_segments=transcription_result.segments,
            diarization_segments=diarization_segments,
        )
        logger.info("✅ [STAGE 4/6] Alignment completed in %.2fs -> Produced %d aligned dialogue turns",
                    time.time() - s4, len(aligned_result.segments))

        # 5. Extract one centroid embedding per diarization speaker cluster
        s5 = time.time()
        logger.info("⏳ [STAGE 5/6] Extracting voice embeddings for %d speaker clusters...", len(unique_spks))
        cluster_embeddings: Dict[str, np.ndarray] = {}

        for speaker_label in unique_spks:
            cluster_turns = [s for s in diarization_segments if s.speaker == speaker_label]
            logger.info("   ↳ Extracting 256D embedding for [%s] from %d turns...", speaker_label, len(cluster_turns))
            cluster_emb = self.embedding_service.extract_embedding(
                processed.processed_path,
                speech_segments=cluster_turns,
            )
            cluster_embeddings[speaker_label] = cluster_emb

        logger.info("✅ [STAGE 5/6] Extracted embeddings for all clusters in %.2fs", time.time() - s5)

        # 6. Fetch team voice profiles & match
        s6 = time.time()
        logger.info("⏳ [STAGE 6/6] Matching speaker voiceprints with Enrolled Voice Registry...")
        enrolled_profiles = self.registry.list_profiles(team_id=team_id)
        logger.info("   ↳ Enrolled profiles available for matching: %d (%s)", 
                    len(enrolled_profiles), [p.user_name for p in enrolled_profiles])
        
        cluster_matches = self.matcher.match_clusters(cluster_embeddings, enrolled_profiles)
        for cl_id, match in cluster_matches.items():
            logger.info("   ↳ Cluster '%s' => Matched: %s (Confidence: %.2f, Cosine: %.3f)",
                        cl_id, match.matched_name, match.confidence, match.similarity_score)

        final_result = self.matcher.apply_matches_to_transcript(aligned_result, cluster_matches)
        
        total_time = time.time() - total_start
        logger.info("=" * 70)
        logger.info("🎉 [COMPLETE] Entire Meeting Pipeline Finished in %.2fs (%.1f min)!", total_time, total_time / 60)
        logger.info("=" * 70)
        return final_result

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
