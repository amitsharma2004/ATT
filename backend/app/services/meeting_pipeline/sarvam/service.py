"""Sarvam Batch Speech-to-Text lifecycle service orchestration."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.audio.processor import AudioProcessor
from backend.app.services.meeting_pipeline.diarization.models import SpeakerSegment
from backend.app.services.meeting_pipeline.sarvam.client import SarvamClientManager
from backend.app.services.meeting_pipeline.sarvam.config import SarvamConfig
from backend.app.services.meeting_pipeline.sarvam.exceptions import (
    SarvamFileUploadError,
    SarvamJobCreationError,
    SarvamJobProcessingError,
    SarvamJobStartError,
    SarvamJobTimeoutError,
    SarvamOutputDownloadError,
    SarvamParserError,
)
from backend.app.services.meeting_pipeline.sarvam.models import (
    SarvamBatchTranscriptResponse,
    SarvamDiarizedSegment,
)
from backend.app.services.meeting_pipeline.sarvam.parser import SarvamOutputParser
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.embedding import SpeakerEmbeddingService
from backend.app.services.meeting_pipeline.speaker.matcher import SpeakerMatcher
from backend.app.services.meeting_pipeline.speaker.models import SpeakerClusterMatch
from backend.app.services.meeting_pipeline.speaker.registry import VoiceRegistry


logger = logging.getLogger(__name__)


class SarvamBatchSTTService:
    """Orchestrates end-to-end Sarvam Batch Speech-to-Text transcription with Diarization.
    
    Workflow:
    1. Preprocess audio via existing AudioProcessor to 16kHz mono PCM WAV
    2. Create batch job with model (saaras:v4) and with_diarization=True
    3. Upload standardized audio file
    4. Start batch job
    5. Asynchronously poll until job completes (with timeout protection)
    6. Download and read output artifacts
    7. Parse into normalized internal transcript response
    """

    def __init__(
        self,
        config: Optional[SarvamConfig] = None,
        settings: Optional[Settings] = None,
        client_manager: Optional[SarvamClientManager] = None,
        audio_processor: Optional[AudioProcessor] = None,
        embedding_service: Optional[SpeakerEmbeddingService] = None,
        registry: Optional[VoiceRegistry] = None,
        matcher: Optional[SpeakerMatcher] = None,
    ) -> None:
        self.settings = settings or get_settings()

        if config is None:
            self.config = SarvamConfig(
                api_key=self.settings.sarvam_api_key,
                model=self.settings.sarvam_model,
                language_code=self.settings.sarvam_language_code,
                mode=self.settings.sarvam_mode,
                with_diarization=self.settings.sarvam_with_diarization,
                poll_interval_seconds=self.settings.sarvam_poll_interval_seconds,
                max_wait_seconds=self.settings.sarvam_max_wait_seconds,
            )
        else:
            self.config = config

        self.client_manager = client_manager or SarvamClientManager(config=self.config, settings=self.settings)
        self.audio_processor = audio_processor or AudioProcessor(settings=self.settings)

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

        self.output_storage_dir = self.settings.storage_dir / "processed" / "sarvam"
        self.output_storage_dir.mkdir(parents=True, exist_ok=True)

    def _sync_create_job(self, client, model: str, mode: str, with_diarization: bool, language_code: str):
        """Create batch STT job on Sarvam."""
        try:
            return client.speech_to_text_job.create_job(
                model=model,
                mode=mode,
                with_diarization=with_diarization,
                language_code=language_code,
            )
        except Exception as exc:
            raise SarvamJobCreationError(f"Failed to create Sarvam batch STT job: {exc}") from exc

    def _sync_upload_files(self, job, file_paths: List[str]):
        """Upload audio files to the created job."""
        try:
            success = job.upload_files(file_paths=file_paths)
            if not success:
                raise SarvamFileUploadError(f"Sarvam upload_files returned failure status for job {job.job_id}")
        except Exception as exc:
            raise SarvamFileUploadError(f"Failed to upload audio to Sarvam: {exc}") from exc

    def _sync_start_job(self, job):
        """Start execution of the batch job."""
        try:
            job.start()
        except Exception as exc:
            raise SarvamJobStartError(f"Failed to start Sarvam batch job {job.job_id}: {exc}") from exc

    def _sync_get_status(self, job):
        """Fetch current job status."""
        return job.get_status()

    def _sync_download_outputs(self, job, output_dir: str):
        """Download output files from completed job."""
        try:
            return job.download_outputs(output_dir=output_dir)
        except Exception as exc:
            raise SarvamOutputDownloadError(f"Failed to download outputs for job {job.job_id}: {exc}") from exc

    async def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        team_id: Optional[str] = None,
        model: Optional[str] = None,
        language_code: Optional[str] = None,
        with_diarization: Optional[bool] = None,
    ) -> SarvamBatchTranscriptResponse:
        """Run complete Sarvam Batch STT transcription and diarization on given audio."""
        raw_path = Path(audio_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {raw_path}")

        # 1. Standardize audio using existing AudioProcessor
        logger.info("Standardizing audio for Sarvam: %s", raw_path.name)
        preprocess_res = self.audio_processor.preprocess(raw_path)
        standardized_path = preprocess_res.processed_path

        # Resolve parameters
        active_model = model or self.config.model
        active_language = language_code or self.config.language_code
        active_diarization = with_diarization if with_diarization is not None else self.config.with_diarization

        # Obtain authenticated client
        client = self.client_manager.get_client()

        t_start = time.monotonic()

        # 2. Create job
        logger.info("Creating Sarvam batch job (model=%s, diarization=%s)...", active_model, active_diarization)
        loop = asyncio.get_running_loop()
        job = await loop.run_in_executor(
            None,
            self._sync_create_job,
            client,
            active_model,
            self.config.mode,
            active_diarization,
            active_language,
        )
        job_id = getattr(job, "job_id", "unknown_job")
        logger.info("Created Sarvam batch job: %s", job_id)

        # 3. Upload file
        logger.info("Uploading audio file to Sarvam job %s...", job_id)
        await loop.run_in_executor(
            None,
            self._sync_upload_files,
            job,
            [str(standardized_path)],
        )

        # 4. Start job
        logger.info("Starting Sarvam job %s...", job_id)
        await loop.run_in_executor(None, self._sync_start_job, job)

        # 5. Poll until completion (non-blocking for asyncio loop)
        poll_interval = self.config.poll_interval_seconds
        max_wait = self.config.max_wait_seconds
        elapsed = 0.0

        logger.info("Waiting for Sarvam job %s to complete (poll_interval=%ds, timeout=%ds)...", job_id, poll_interval, max_wait)

        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed = time.monotonic() - t_start

            status_res = await loop.run_in_executor(None, self._sync_get_status, job)
            raw_state = getattr(status_res, "job_state", "")
            state = str(raw_state).lower() if raw_state else ""

            logger.info("Sarvam job %s status: %s (elapsed: %.1fs)", job_id, state, elapsed)

            if state in ("completed", "successful", "success", "done"):
                break
            if state in ("failed", "failure", "error"):
                error_msg = getattr(status_res, "error", f"Job failed with state: {state}")
                raise SarvamJobProcessingError(f"Sarvam batch job {job_id} failed: {error_msg}")

        if elapsed >= max_wait:
            raise SarvamJobTimeoutError(
                f"Sarvam batch job {job_id} timed out after {round(elapsed, 1)}s (limit: {max_wait}s)"
            )

        # 6. Download outputs to predictable job folder
        job_output_dir = self.output_storage_dir / str(job_id)
        job_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading Sarvam outputs to %s...", job_output_dir)
        await loop.run_in_executor(None, self._sync_download_outputs, job, str(job_output_dir))

        # 7. Locate output JSON
        json_files = list(job_output_dir.glob("*.json"))
        if not json_files:
            # Check subdirectories
            json_files = list(job_output_dir.rglob("*.json"))

        if not json_files:
            raise SarvamOutputDownloadError(
                f"No JSON output artifact found for Sarvam job {job_id} in {job_output_dir}"
            )

        # Read the first json artifact
        output_file = json_files[0]
        try:
            raw_json = json.loads(output_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SarvamParserError(f"Failed to read or parse JSON artifact {output_file.name}: {exc}") from exc

        # 8. Parse into standardized domain model
        total_elapsed = round(time.monotonic() - t_start, 2)
        response = SarvamOutputParser.parse_job_output(
            raw_output=raw_json,
            fallback_model=active_model,
            fallback_language=active_language,
            job_id=job_id,
            elapsed_seconds=total_elapsed,
        )

        logger.info(
            "Parsed Sarvam transcript successfully: %d segments, %d speakers, audio duration: %s",
            len(response.segments),
            response.total_speakers,
            response.audio_duration,
        )

        # 9. Perform Speaker Embedding Extraction + VoiceRegistry Matching if segments and profiles exist
        enrolled_profiles = self.registry.list_profiles(team_id=team_id)
        if response.segments and enrolled_profiles:
            try:
                unique_speakers = sorted(list(set(s.speaker for s in response.segments)))
                cluster_embeddings: Dict[str, Any] = {}

                for spk_label in unique_speakers:
                    # Collect all interval speech segments for this speaker cluster
                    cluster_turns = [
                        SpeakerSegment(
                            speaker=s.speaker,
                            start=s.start,
                            end=s.end,
                            duration=round(s.end - s.start, 4),
                        )
                        for s in response.segments
                        if s.speaker == spk_label and (s.end - s.start) > 0.05
                    ]
                    if cluster_turns:
                        cluster_emb = await loop.run_in_executor(
                            None,
                            self.embedding_service.extract_embedding,
                            standardized_path,
                            cluster_turns,
                        )
                        cluster_embeddings[spk_label] = cluster_emb

                if cluster_embeddings:
                    cluster_matches = self.matcher.match_clusters(cluster_embeddings, enrolled_profiles)
                    match_map: Dict[str, SpeakerClusterMatch] = {
                        m.cluster_speaker: m for m in cluster_matches
                    }

                    named_segments: List[SarvamDiarizedSegment] = []
                    for seg in response.segments:
                        match = match_map.get(seg.speaker)
                        if match and match.matched_name != "UNKNOWN":
                            s_name = match.matched_name
                            u_id = match.matched_user_id
                            m_conf = match.confidence
                        else:
                            s_name = seg.speaker
                            u_id = None
                            m_conf = 0.0

                        named_segments.append(
                            SarvamDiarizedSegment(
                                speaker=seg.speaker,
                                speaker_name=s_name,
                                user_id=u_id,
                                start=seg.start,
                                end=seg.end,
                                text=seg.text,
                                match_confidence=m_conf,
                            )
                        )

                    return SarvamBatchTranscriptResponse(
                        engine=response.engine,
                        model=response.model,
                        language=response.language,
                        audio_duration=response.audio_duration,
                        total_speakers=response.total_speakers,
                        speaker_clusters=cluster_matches,
                        segments=named_segments,
                        metadata=response.metadata,
                    )
            except Exception as match_err:
                logger.warning("Speaker voiceprint matching failed on Sarvam segments, returning anonymous: %s", match_err)

        return response

