"""
SpeakerMatcher Service
========================
backend/app/services/meeting_pipeline/speaker/matcher.py

Takes per-cluster centroid embeddings from a meeting (SPEAKER_00,
SPEAKER_01, ...) and matches each cluster against enrolled team-member
voice profiles in the VoiceRegistry.

Stages:
    1. Pairwise Cosine Similarity Engine   (best-of-enrollment-samples,
                                             not just the profile centroid)
    2. Cohort Score Normalization          (cancels session-level bias —
                                             noisy mic, different room, etc.
                                             that dampens ALL similarities
                                             in a given meeting)
    3. Multi-Tier Confidence Evaluator     (raw-floor + normalized z-score)
    4. Global Exclusive Hungarian/Greedy Assignment (with a margin check
                                             against the runner-up)
    5. Transcript Re-labeling              (SPEAKER_00 -> Amit Sharma, etc.)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.services.meeting_pipeline.alignment.aligner import UNKNOWN_SPEAKER
from backend.app.services.meeting_pipeline.alignment.models import AlignedTranscriptResult
from backend.app.services.meeting_pipeline.speaker.config import SpeakerConfig
from backend.app.services.meeting_pipeline.speaker.models import (
    ConfidenceTier,
    MeetingTranscriptionResult,
    NamedTranscriptSegment,
    SpeakerClusterMatch,
    SpeakerMatchCandidate,
    VoiceProfile,
)
from backend.app.services.meeting_pipeline.speaker.registry import VoiceRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence thresholds & floors
# ---------------------------------------------------------------------------
HIGH_RAW_FLOOR = 0.75
MEDIUM_RAW_FLOOR = 0.45

# Cohort z-score thresholds (how far above the rest of the candidates this
# score stands, in std-devs)
HIGH_Z_THRESHOLD = 1.5
MEDIUM_Z_THRESHOLD = 0.8

# Minimum margin (raw cosine) the winning candidate must have over the runner-up
MIN_WINNER_MARGIN = 0.03


@dataclass
class MatchCandidate:
    person_id: str
    display_name: str
    raw_similarity: float
    normalized_score: float
    tier: ConfidenceTier


@dataclass
class SpeakerAssignment:
    cluster_label: str
    person_id: Optional[str]
    display_name: str
    raw_similarity: float
    normalized_score: float
    tier: ConfidenceTier


# ---------------------------------------------------------------------------
# 1. Pairwise Cosine Similarity Engine — best-of-enrollment-samples
# ---------------------------------------------------------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_sample_similarity(cluster_centroid: np.ndarray, profile: VoiceProfile) -> float:
    """
    Compares the meeting cluster against EVERY enrollment sample for this
    person (not just their averaged centroid) and returns the best score.
    """
    sims = [cosine_similarity(cluster_centroid, emb) for emb in profile.embeddings]
    if not sims:
        return cosine_similarity(cluster_centroid, profile.centroid)
    return float(max(sims))


def build_similarity_matrix(
    cluster_centroids: Dict[str, np.ndarray],
    profiles: List[VoiceProfile],
) -> Dict[str, Dict[str, float]]:
    """
    Returns nested dict: {cluster_label: {person_id: raw_similarity}}
    raw_similarity = best score against any of that person's enrollment samples.
    """
    matrix: Dict[str, Dict[str, float]] = {}
    for cluster_label, centroid in cluster_centroids.items():
        matrix[cluster_label] = {
            profile.person_id: best_sample_similarity(centroid, profile)
            for profile in profiles
        }
    return matrix


# ---------------------------------------------------------------------------
# 2. Cohort Score Normalization
# ---------------------------------------------------------------------------
def normalize_cluster_scores(sims: Dict[str, float]) -> Dict[str, float]:
    """
    For a single cluster's similarity scores against every enrolled person,
    convert each raw score into a z-score relative to the OTHER candidates
    ("leave-one-out" cohort normalization).
    """
    person_ids = list(sims.keys())
    if len(person_ids) < 2:
        return {pid: 0.0 for pid in person_ids}

    values = np.array([sims[pid] for pid in person_ids], dtype=np.float64)
    normalized: Dict[str, float] = {}
    for i, pid in enumerate(person_ids):
        mask = np.ones(len(values), dtype=bool)
        mask[i] = False
        others = values[mask]
        mean, std = float(others.mean()), float(others.std())
        std = std if std > 1e-6 else 1e-6
        normalized[pid] = (values[i] - mean) / std
    return normalized


# ---------------------------------------------------------------------------
# 3. Multi-Tier Confidence Evaluator (raw floor + normalized z-score)
# ---------------------------------------------------------------------------
def classify_confidence(raw_similarity: float, normalized_score: float) -> ConfidenceTier:
    if raw_similarity >= HIGH_RAW_FLOOR:
        return ConfidenceTier.HIGH
    if raw_similarity >= MEDIUM_RAW_FLOOR and normalized_score >= HIGH_Z_THRESHOLD:
        return ConfidenceTier.HIGH
    if raw_similarity >= MEDIUM_RAW_FLOOR and normalized_score >= MEDIUM_Z_THRESHOLD:
        return ConfidenceTier.MEDIUM
    if raw_similarity >= 0.65:  # legacy absolute-threshold safety net
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.UNKNOWN


def rank_candidates(
    cluster_label: str,
    similarity_matrix: Dict[str, Dict[str, float]],
    profiles_by_id: Dict[str, VoiceProfile],
) -> List[MatchCandidate]:
    raw_sims = similarity_matrix.get(cluster_label, {})
    normalized = normalize_cluster_scores(raw_sims)

    candidates = [
        MatchCandidate(
            person_id=pid,
            display_name=profiles_by_id[pid].display_name,
            raw_similarity=raw_sims[pid],
            normalized_score=normalized[pid],
            tier=classify_confidence(raw_sims[pid], normalized[pid]),
        )
        for pid in raw_sims
    ]
    candidates.sort(key=lambda c: (c.raw_similarity, c.normalized_score), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 4. Global Exclusive Hungarian/Greedy Assignment (+ winner margin check)
# ---------------------------------------------------------------------------
def _hungarian_assignment(
    cluster_labels: List[str],
    person_ids: List[str],
    similarity_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Optimal one-to-one assignment maximizing total similarity using scipy linear_sum_assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _greedy_assignment(cluster_labels, person_ids, similarity_matrix)

    n, m = len(cluster_labels), len(person_ids)
    cost = np.ones((n, m), dtype=np.float32)  # cost = 1 - similarity (minimize)
    for i, cl in enumerate(cluster_labels):
        for j, pid in enumerate(person_ids):
            cost[i, j] = 1.0 - similarity_matrix.get(cl, {}).get(pid, 0.0)

    row_idx, col_idx = linear_sum_assignment(cost)
    assignment = {cluster_labels[i]: person_ids[j] for i, j in zip(row_idx, col_idx)}
    return assignment


def _greedy_assignment(
    cluster_labels: List[str],
    person_ids: List[str],
    similarity_matrix: Dict[str, Dict[str, float]],
) -> Dict[str, str]:
    """Fallback: repeatedly pick the globally best remaining (cluster, person) pair."""
    pairs = []
    for cl in cluster_labels:
        for pid in person_ids:
            sim = similarity_matrix.get(cl, {}).get(pid, 0.0)
            pairs.append((sim, cl, pid))
    pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_clusters, assigned_people = set(), set()
    assignment: Dict[str, str] = {}
    for sim, cl, pid in pairs:
        if cl in assigned_clusters or pid in assigned_people:
            continue
        assignment[cl] = pid
        assigned_clusters.add(cl)
        assigned_people.add(pid)
    return assignment


def exclusive_assign(
    cluster_centroids: Dict[str, np.ndarray],
    profiles: List[VoiceProfile],
    similarity_matrix: Dict[str, Dict[str, float]],
) -> List[SpeakerAssignment]:
    """
    Assigns each meeting cluster to at most one enrolled member using Hungarian
    optimization with cohort z-scores and margin validation against the runner-up.
    """
    cluster_labels = list(cluster_centroids.keys())
    profiles_by_id = {p.person_id: p for p in profiles}
    person_ids = list(profiles_by_id.keys())

    raw_assignment: Dict[str, str] = {}
    if cluster_labels and person_ids:
        raw_assignment = _hungarian_assignment(cluster_labels, person_ids, similarity_matrix)

    results: List[SpeakerAssignment] = []
    for cl in cluster_labels:
        candidates = rank_candidates(cl, similarity_matrix, profiles_by_id)
        pid = raw_assignment.get(cl)

        if pid is None or not candidates:
            results.append(SpeakerAssignment(cl, None, cl, 0.0, 0.0, ConfidenceTier.UNKNOWN))
            continue

        winner = next((c for c in candidates if c.person_id == pid), None)
        runner_up = next((c for c in candidates if c.person_id != pid), None)

        if winner is None:
            results.append(SpeakerAssignment(cl, None, cl, 0.0, 0.0, ConfidenceTier.UNKNOWN))
            continue

        margin_ok = (
            runner_up is None
            or (winner.raw_similarity - runner_up.raw_similarity) >= MIN_WINNER_MARGIN
        )

        if winner.tier != ConfidenceTier.UNKNOWN and margin_ok:
            results.append(
                SpeakerAssignment(
                    cluster_label=cl,
                    person_id=pid,
                    display_name=winner.display_name,
                    raw_similarity=winner.raw_similarity,
                    normalized_score=winner.normalized_score,
                    tier=winner.tier,
                )
            )
        else:
            results.append(
                SpeakerAssignment(
                    cluster_label=cl,
                    person_id=None,
                    display_name=cl,
                    raw_similarity=winner.raw_similarity,
                    normalized_score=winner.normalized_score,
                    tier=ConfidenceTier.UNKNOWN,
                )
            )

    return results


# ---------------------------------------------------------------------------
# 5. Transcript Re-labeling
# ---------------------------------------------------------------------------
def relabel_transcript(
    transcript_segments: List[dict],
    assignments: List[SpeakerAssignment],
) -> List[dict]:
    label_map = {a.cluster_label: a.display_name for a in assignments}
    relabeled = []
    for seg in transcript_segments:
        new_seg = dict(seg)
        new_seg["speaker"] = label_map.get(seg["speaker"], seg["speaker"])
        relabeled.append(new_seg)
    return relabeled


# ---------------------------------------------------------------------------
# SpeakerMatcher Service Class
# ---------------------------------------------------------------------------
class SpeakerMatcher:
    """Matches meeting speaker clusters against enrolled team voiceprints."""

    def __init__(
        self,
        config: Optional[SpeakerConfig] = None,
        registry: Optional[VoiceRegistry] = None,
    ) -> None:
        self.config = config or SpeakerConfig()
        self.registry = registry

    def match_clusters(
        self,
        cluster_embeddings: Dict[str, np.ndarray],
        enrolled_profiles: List[VoiceProfile],
    ) -> List[SpeakerClusterMatch]:
        """Matches cluster embeddings against enrolled profiles."""
        if not cluster_embeddings:
            return []

        if not enrolled_profiles:
            return [
                SpeakerClusterMatch(
                    cluster_speaker=cluster,
                    matched_user_id=None,
                    matched_name=UNKNOWN_SPEAKER,
                    confidence=0.0,
                    cosine_similarity=0.0,
                    normalized_score=0.0,
                    match_status="UNKNOWN",
                    candidates=[],
                )
                for cluster in cluster_embeddings.keys()
            ]

        profiles_by_id = {p.person_id: p for p in enrolled_profiles}
        similarity_matrix = build_similarity_matrix(cluster_embeddings, enrolled_profiles)
        assignments = exclusive_assign(cluster_embeddings, enrolled_profiles, similarity_matrix)

        assignment_map = {a.cluster_label: a for a in assignments}
        matches: List[SpeakerClusterMatch] = []

        for cluster in cluster_embeddings.keys():
            candidates = rank_candidates(cluster, similarity_matrix, profiles_by_id)
            cand_models = [
                SpeakerMatchCandidate(
                    user_id=c.person_id,
                    name=c.display_name,
                    cosine_similarity=round(c.raw_similarity, 4),
                    normalized_score=round(c.normalized_score, 4),
                    tier=c.tier.value,
                )
                for c in candidates
            ]

            assign = assignment_map.get(cluster)
            if assign and assign.tier != ConfidenceTier.UNKNOWN and assign.person_id:
                conf = min(1.0, max(0.0, round(assign.raw_similarity, 2)))
                matches.append(
                    SpeakerClusterMatch(
                        cluster_speaker=cluster,
                        matched_user_id=assign.person_id,
                        matched_name=assign.display_name,
                        confidence=conf,
                        cosine_similarity=round(assign.raw_similarity, 4),
                        normalized_score=round(assign.normalized_score, 4),
                        match_status=assign.tier.value,
                        candidates=cand_models,
                    )
                )
            else:
                top_sim = candidates[0].raw_similarity if candidates else 0.0
                top_z = candidates[0].normalized_score if candidates else 0.0
                matches.append(
                    SpeakerClusterMatch(
                        cluster_speaker=cluster,
                        matched_user_id=None,
                        matched_name=UNKNOWN_SPEAKER,
                        confidence=0.0,
                        cosine_similarity=round(top_sim, 4),
                        normalized_score=round(top_z, 4),
                        match_status="UNKNOWN",
                        candidates=cand_models,
                    )
                )

        return matches

    def match_meeting(
        self,
        cluster_centroids: Dict[str, np.ndarray],
        transcript_segments: Optional[List[dict]] = None,
    ) -> dict:
        """Helper method when registry is injected into matcher directly."""
        if not self.registry:
            raise RuntimeError("VoiceRegistry is required for match_meeting()")
        profiles = self.registry.all_profiles()
        similarity_matrix = build_similarity_matrix(cluster_centroids, profiles)
        assignments = exclusive_assign(cluster_centroids, profiles, similarity_matrix)

        output: Dict[str, Any] = {
            "assignments": assignments,
            "similarity_matrix": similarity_matrix,
        }

        if transcript_segments is not None:
            output["relabeled_transcript"] = relabel_transcript(transcript_segments, assignments)

        return output

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
                spk_name = seg.speaker
                uid = None
                m_conf = 0.0

            # When Whisper task="translate", seg.text is already translated to English by Whisper
            eng_text = seg.text
            formatted_line = f"{spk_name}: {eng_text}"

            named_segments.append(
                NamedTranscriptSegment(
                    speaker_name=spk_name,
                    speaker_cluster=seg.speaker,
                    user_id=uid,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    translated_text=eng_text,
                    english_line=formatted_line,
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
