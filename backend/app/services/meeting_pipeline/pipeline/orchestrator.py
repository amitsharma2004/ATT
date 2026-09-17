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
        from backend.app.services.meeting_pipeline.alignment.forced_aligner import Wav2VecForcedAligner
        self.forced_aligner = Wav2VecForcedAligner(device=self.settings.compute_device)

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
        processed = self.audio_processor.preprocess(Path(audio_path))
        logger.info("[STAGE 1/7] Audio Preprocessing   | %.2fs | File: %s (Duration: %.1fs)", 
                    time.time() - s1, processed.processed_path.name, processed.duration_seconds)

        # 2. Pyannote speaker diarization
        s2 = time.time()
        diarization_segments = self.diarizer.diarize(
            processed.processed_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
        )
        unique_spks = sorted(list(set(s.speaker for s in diarization_segments)))
        logger.info("[STAGE 2/7] Speaker Diarization   | %.2fs | Found %d segments across %d speaker(s): %s",
                    time.time() - s2, len(diarization_segments), len(unique_spks), ", ".join(unique_spks))

        # 3. Whisper speech-to-text
        s3 = time.time()
        transcription_result = self.transcriber.transcribe(
            processed.processed_path,
            language=language,
        )
        logger.info("[STAGE 3/7] Whisper STT (Large-v3)| %.2fs | Transcribed %d segments (Language: %s)",
                    time.time() - s3, len(transcription_result.segments), transcription_result.language)

        # 4. Temporal timestamp alignment (using Wav2Vec2 phoneme/word alignment with fallback)
        s4 = time.time()
        try:
            aligned_result = self.forced_aligner.align_and_attribute_speakers(
                audio_path=processed.processed_path,
                transcript_segments=transcription_result.segments,
                diarization_segments=diarization_segments,
            )
        except Exception as align_err:
            logger.warning("[STAGE 4/7] Forced alignment fallback triggered: %s", align_err)
            aligned_result = self.aligner.align(
                transcript_segments=transcription_result.segments,
                diarization_segments=diarization_segments,
            )
        logger.info("[STAGE 4/7] Timestamp Alignment   | %.2fs | Produced %d dialogue turns",
                    time.time() - s4, len(aligned_result.segments))

        # 5. Extract one centroid embedding per diarization speaker cluster
        s5 = time.time()
        cluster_embeddings: Dict[str, np.ndarray] = {}
        for speaker_label in unique_spks:
            cluster_turns = [s for s in diarization_segments if s.speaker == speaker_label]
            cluster_emb = self.embedding_service.extract_embedding(
                processed.processed_path,
                speech_segments=cluster_turns,
            )
            cluster_embeddings[speaker_label] = cluster_emb
        logger.info("[STAGE 5/7] Voice Embeddings (256D)| %.2fs | Extracted for %d speaker clusters", 
                    time.time() - s5, len(unique_spks))

        # 6. Fetch team voice profiles & match
        s6 = time.time()
        enrolled_profiles = self.registry.list_profiles(team_id=team_id)
        cluster_matches = self.matcher.match_clusters(cluster_embeddings, enrolled_profiles)
        match_summary = ", ".join([f"{m.cluster_speaker}->{m.matched_name}({m.match_status})" for m in cluster_matches])
        logger.info("[STAGE 6/8] Voiceprint Matching   | %.2fs | %s", time.time() - s6, match_summary)

        final_result = self.matcher.apply_matches_to_transcript(aligned_result, cluster_matches)

        # 7. Speaker-Aware Chunked Translation (Memory Preserved)
        # DISABLED for this push: Whisper now runs with task="translate" (see
        # whisper_service.py) and produces English text directly, so this
        # extra local-LLM chunked-translation pass isn't needed for now.
        # Left in place (not deleted) — uncomment to re-enable when ready.
        # The translate/ package (chunked_translate.py, job_manager.py) and the
        # /api/translation/jobs routes in pipeline.py are the matching other
        # half of this feature and are commented out for the same reason.
        s7 = time.time()
        translation_meta: Dict[str, object] = {"translation_status": "disabled"}
        logger.info("[STAGE 7/8] Chunked Translation   | disabled (Whisper task=translate handles English output)")
        # if final_result.segments:
        #     try:
        #         from backend.app.services.meeting_pipeline.translate import (
        #             SpeakerTranscriptSegment,
        #             translate_transcript,
        #         )
        #         from backend.app.services.meeting_pipeline.translation_service import indic_translation_service
        #
        #         typed_segs = [
        #             SpeakerTranscriptSegment(
        #                 start=seg.start,
        #                 end=seg.end,
        #                 text=seg.text,
        #                 speaker=seg.speaker_name,
        #             )
        #             for seg in final_result.segments
        #         ]
        #
        #         def translate_fn(prompt: str) -> str:
        #             return indic_translation_service.generate_raw_completion(
        #                 prompt=prompt, max_new_tokens=4096, temperature=0.0, timeout_s=120.0
        #             )
        #
        #         translated_segs, memory, failed_chunks = translate_transcript(
        #             segments=typed_segs,
        #             translate_fn=translate_fn,
        #             target_language="English",
        #             max_input_tokens_per_chunk=1500,
        #         )
        #
        #         # Update translated text and formatted line on final segments
        #         updated_segments = []
        #         for orig_seg, tr_seg in zip(final_result.segments, translated_segs):
        #             updated_seg = orig_seg.model_copy(
        #                 update={
        #                     "translated_text": tr_seg.text,
        #                     "english_line": f"{orig_seg.speaker_name}: {tr_seg.text}",
        #                 }
        #             )
        #             updated_segments.append(updated_seg)
        #
        #         final_result = final_result.model_copy(update={"segments": updated_segments})
        #         translation_status = "partial" if failed_chunks else "success"
        #         translation_meta = {
        #             "translation_status": translation_status,
        #             "translation_failed_chunks": failed_chunks,
        #             "translation_glossary_size": len(memory.glossary),
        #         }
        #         logger.info("[STAGE 7/8] Chunked Translation   | %.2fs | Translated %d segments (Memory Glossary: %d entries, failed chunks: %s)",
        #                     time.time() - s7, len(updated_segments), len(memory.glossary), failed_chunks or "none")
        #     except Exception as tr_err:
        #         translation_meta = {"translation_status": "failed", "translation_error": str(tr_err)}
        #         logger.warning("[STAGE 7/8] Chunked translation skipped/fallback: %s", tr_err)
        # else:
        #     logger.info("[STAGE 7/8] Chunked Translation   | 0.00s | No segments to translate")

        # 8. Generate Meeting Summary using Local Qwen
        meeting_summary = None
        summary_status = "skipped"
        summary_meta: Dict[str, object] = {}
        if final_result.segments:
            s8 = time.time()
            dialogue_lines = [
                seg.english_line or f"{seg.speaker_name}: {seg.text}"
                for seg in final_result.segments
            ]
            try:
                from backend.app.services.meeting_pipeline.translation_service import indic_translation_service
                sum_res = indic_translation_service.summarize_meeting(dialogue_lines, timeout_s=120.0)
                meeting_summary = sum_res.get("summary_markdown")
                if sum_res.get("error"):
                    summary_status = "failed"
                    summary_meta = {"summary_status": summary_status, "summary_error": sum_res["error"]}
                else:
                    summary_status = "success" if meeting_summary else "empty"
                    summary_meta = {"summary_status": summary_status}
                logger.info("[STAGE 8/8] Qwen Meeting Summary  | %.2fs | Status: %s",
                            time.time() - s8, summary_status.upper())
            except Exception as sum_err:
                summary_status = "failed"
                summary_meta = {"summary_status": summary_status, "summary_error": str(sum_err)}
                logger.warning("[STAGE 8/8] Summary generation skipped: %s", sum_err)

        if meeting_summary:
            final_result = final_result.model_copy(update={"summary": meeting_summary})

        final_result = final_result.model_copy(
            update={"metadata": {**final_result.metadata, **translation_meta, **summary_meta}}
        )

        total_time = time.time() - total_start
        logger.info("=" * 78)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY: Total Duration: %.2fs (%.1f min) | Output Segments: %d", 
                    total_time, total_time / 60, len(final_result.segments))
        logger.info("=" * 78)
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
