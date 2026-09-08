"""Speaker Matcher computing cosine similarity and global exclusive assignment."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from backend.app.services.meeting_pipeline.alignment.aligner import UNKNOWN_SPEAKER
from backend.app.services.meeting_pipeline.alignment.models import (
    AlignedTranscriptResult,
    AlignedTranscriptSegment,
)
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.models import (
    MeetingTranscriptionResult,
    NamedTranscriptSegment,
    SpeakerClusterMatch,
    SpeakerMatchCandidate,
    VoiceProfile,
)

logger = logging.getLogger(__name__)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors: (A . B) / (||A|| * ||B||)."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    dot = np.dot(v1, v2)
    return float(dot / (norm1 * norm2))


class SpeakerMatcher:
    """Matches meeting speaker clusters (SPEAKER_00, SPEAKER_01...) against enrolled team voiceprints.

    Features:
    - Multi-tier confidence evaluation (HIGH, MEDIUM, UNKNOWN)
    - Global exclusive matching (one enrolled identity cannot be assigned to multiple clusters in one meeting)
    - Re-labels aligned transcript segments with team member names
    """

    def __init__(self, config: Optional[SpeakerConfig] = None) -> None:
        self.config = config or SpeakerConfig()

    def match_clusters(
        self,
        cluster_embeddings: Dict[str, np.ndarray],
        enrolled_profiles: List[VoiceProfile],
    ) -> List[SpeakerClusterMatch]:
        """Match each speaker cluster embedding against the registry profiles.

        Args:
            cluster_embeddings: Mapping of cluster label (e.g. 'SPEAKER_00') to its embedding vector.
            enrolled_profiles: List of enrolled VoiceProfile objects.

        Returns:
            List of SpeakerClusterMatch decisions.
        """
        if not cluster_embeddings:
            return []

        if not enrolled_profiles:
            # No enrolled team members: all clusters remain UNKNOWN
            return [
                SpeakerClusterMatch(
                    cluster_speaker=cluster,
                    matched_user_id=None,
                    matched_name=UNKNOWN_SPEAKER,
                    confidence=0.0,
                    cosine_similarity=0.0,
                    match_status="UNKNOWN",
                    candidates=[],
                )
                for cluster in cluster_embeddings.keys()
            ]

        # 1. Build candidate evaluation list and global pairwise scores
        all_candidate_evaluations: Dict[str, List[SpeakerMatchCandidate]] = {}
        pairwise_scores: List[Tuple[float, str, VoiceProfile]] = []

        for cluster, emb in cluster_embeddings.items():
            candidates: List[SpeakerMatchCandidate] = []
            for profile in enrolled_profiles:
                prof_emb = np.array(profile.embedding, dtype=np.float32)
                sim = cosine_similarity(emb, prof_emb)
                candidates.append(
                    SpeakerMatchCandidate(
                        user_id=profile.user_id,
                        name=profile.name,
                        cosine_similarity=round(sim, 4),
                    )
                )
                pairwise_scores.append((sim, cluster, profile))

            candidates.sort(key=lambda c: c.cosine_similarity, reverse=True)
            all_candidate_evaluations[cluster] = candidates

        assigned_clusters: Dict[str, SpeakerClusterMatch] = {}
        used_user_ids: Set[str] = set()

        # 2. Global exclusive assignment
        if self.config.enforce_exclusive_assignment:
            pairwise_scores.sort(key=lambda x: x[0], reverse=True)

            for sim, cluster, profile in pairwise_scores:
                if cluster in assigned_clusters:
                    continue
                if profile.user_id in used_user_ids:
                    continue

                status, conf = self._evaluate_confidence(sim)
                if status != "UNKNOWN":
                    assigned_clusters[cluster] = SpeakerClusterMatch(
                        cluster_speaker=cluster,
                        matched_user_id=profile.user_id,
                        matched_name=profile.name,
                        confidence=conf,
                        cosine_similarity=round(sim, 4),
                        match_status=status,
                        candidates=all_candidate_evaluations[cluster],
                    )
                    used_user_ids.add(profile.user_id)

        # 3. For any remaining clusters, find best available candidate or UNKNOWN
        for cluster, candidates in all_candidate_evaluations.items():
            if cluster in assigned_clusters:
                continue

            matched = False
            for cand in candidates:
                if self.config.enforce_exclusive_assignment and cand.user_id in used_user_ids:
                    continue

                status, conf = self._evaluate_confidence(cand.cosine_similarity)
                if status != "UNKNOWN":
                    assigned_clusters[cluster] = SpeakerClusterMatch(
                        cluster_speaker=cluster,
                        matched_user_id=cand.user_id,
                        matched_name=cand.name,
                        confidence=conf,
                        cosine_similarity=cand.cosine_similarity,
                        match_status=status,
                        candidates=candidates,
                    )
                    if self.config.enforce_exclusive_assignment:
                        used_user_ids.add(cand.user_id)
                    matched = True
                    break

            if not matched:
                top_sim = candidates[0].cosine_similarity if candidates else 0.0
                assigned_clusters[cluster] = SpeakerClusterMatch(
                    cluster_speaker=cluster,
                    matched_user_id=None,
                    matched_name=UNKNOWN_SPEAKER,
                    confidence=0.0,
                    cosine_similarity=top_sim,
                    match_status="UNKNOWN",
                    candidates=candidates,
                )

        return [assigned_clusters[c] for c in cluster_embeddings.keys()]

    def _evaluate_confidence(self, similarity: float) -> Tuple[str, float]:
        """Classify similarity score into confidence tier and scaled confidence value."""
        if similarity >= self.config.high_confidence_threshold:
            conf = min(1.0, round(similarity, 2))
            return ("HIGH_CONFIDENCE", conf)
        elif similarity >= self.config.medium_confidence_threshold:
            conf = min(1.0, round(similarity, 2))
            return ("MEDIUM_CONFIDENCE", conf)
        else:
            return ("UNKNOWN", 0.0)

    def apply_matches_to_transcript(
        self,
        aligned_result: AlignedTranscriptResult,
        cluster_matches: List[SpeakerClusterMatch],
    ) -> MeetingTranscriptionResult:
        """Map resolved team member names back onto aligned transcript segments."""
        match_map: Dict[str, SpeakerClusterMatch] = {
            m.cluster_speaker: m for m in cluster_matches
        }

        named_segments: List[NamedTranscriptSegment] = []
        for seg in aligned_result.segments:
            match = match_map.get(seg.speaker)
            if match and match.matched_name != UNKNOWN_SPEAKER:
                spk_name = match.matched_name
                uid = match.matched_user_id
                m_conf = match.confidence
            else:
                spk_name = seg.speaker  # Retains 'SPEAKER_XX' or 'UNKNOWN'
                uid = None
                m_conf = 0.0

            named_segments.append(
                NamedTranscriptSegment(
                    speaker_name=spk_name,
                    speaker_cluster=seg.speaker,
                    user_id=uid,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    alignment_confidence=seg.alignment_confidence,
                    match_confidence=m_conf,
                )
            )

        return MeetingTranscriptionResult(
            speaker_clusters=cluster_matches,
            segments=named_segments,
            audio_duration=aligned_result.audio_duration,
            total_speakers=aligned_result.total_speakers,
        )
