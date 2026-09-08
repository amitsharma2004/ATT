"""Timestamp alignment engine matching Whisper transcripts with Pyannote diarization."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from backend.app.services.meeting_pipeline.alignment.exceptions import InvalidTimestampError
from backend.app.services.meeting_pipeline.alignment.models import (
    AlignedTranscriptResult,
    AlignedTranscriptSegment,
    SpeakerCandidateEvidence,
)
from backend.app.services.meeting_pipeline.diarization.models import SpeakerSegment
from backend.app.services.meeting_pipeline.stt.models import TranscriptionSegment

logger = logging.getLogger(__name__)

UNKNOWN_SPEAKER = "UNKNOWN"


class TimestampAligner:
    """Combines Whisper transcription segments and Pyannote speaker segments into an aligned transcript.

    Uses mathematically exact interval intersection, candidate dominance evaluation,
    and conservative same-speaker turn concatenation.
    """

    def __init__(
        self,
        min_overlap_ratio: float = 0.25,
        dominance_threshold: float = 0.60,
        dominance_ratio_margin: float = 0.15,
        merge_gap_s: float = 0.50,
        enable_merging: bool = True,
    ) -> None:
        """Args:

        min_overlap_ratio: Minimum overlap ratio of Whisper segment to consider speaker active.
        dominance_threshold: If single candidate has >= this ratio, candidate is dominant.
        dominance_ratio_margin: Best candidate ratio must exceed 2nd best by at least this margin.
        merge_gap_s: Max time gap in seconds between consecutive same-speaker segments to merge.
        enable_merging: Whether to merge adjacent same-speaker transcript segments.
        """
        self.min_overlap_ratio = min_overlap_ratio
        self.dominance_threshold = dominance_threshold
        self.dominance_ratio_margin = dominance_ratio_margin
        self.merge_gap_s = merge_gap_s
        self.enable_merging = enable_merging

    def align(
        self,
        transcript_segments: List[TranscriptionSegment],
        diarization_segments: List[SpeakerSegment],
    ) -> AlignedTranscriptResult:
        """Align transcript segments against diarization segments.

        Args:
            transcript_segments: Chronological list of Whisper segments.
            diarization_segments: Chronological list of Pyannote turns.

        Returns:
            AlignedTranscriptResult containing chronologically ordered AlignedTranscriptSegments.
        """
        if not transcript_segments:
            return AlignedTranscriptResult(segments=[], total_speakers=0, audio_duration=0.0)

        # Validate timestamps
        for seg in transcript_segments:
            if seg.start < 0 or seg.end <= seg.start:
                raise InvalidTimestampError(
                    f"Invalid transcript segment timestamp: start={seg.start}, end={seg.end}"
                )

        aligned_segments: List[AlignedTranscriptSegment] = []

        for t_seg in transcript_segments:
            aligned_seg = self._align_single_segment(t_seg, diarization_segments)
            aligned_segments.append(aligned_seg)

        # Conservative adjacent same-speaker segment merging
        if self.enable_merging:
            aligned_segments = self._merge_adjacent_aligned_segments(
                aligned_segments,
                diarization_segments=diarization_segments,
            )

        # Calculate metadata
        known_speakers = set(s.speaker for s in aligned_segments if s.speaker != UNKNOWN_SPEAKER)
        total_duration = aligned_segments[-1].end if aligned_segments else 0.0

        return AlignedTranscriptResult(
            segments=aligned_segments,
            total_speakers=len(known_speakers),
            audio_duration=round(total_duration, 3),
        )

    def _align_single_segment(
        self,
        transcript_seg: TranscriptionSegment,
        diarization_segments: List[SpeakerSegment],
    ) -> AlignedTranscriptSegment:
        """Determine assigned speaker for a single Whisper segment based on temporal overlap."""
        w_start = transcript_seg.start
        w_end = transcript_seg.end
        w_duration = max(1e-6, w_end - w_start)

        # Accumulate overlap duration per candidate speaker
        speaker_overlaps: Dict[str, float] = defaultdict(float)

        for d_seg in diarization_segments:
            if d_seg.end <= w_start:
                continue
            if d_seg.start >= w_end:
                continue

            # Mathematically exact interval intersection
            overlap_start = max(w_start, d_seg.start)
            overlap_end = min(w_end, d_seg.end)
            overlap_dur = max(0.0, overlap_end - overlap_start)

            if overlap_dur > 0.0:
                speaker_overlaps[d_seg.speaker] += overlap_dur

        if not speaker_overlaps:
            return AlignedTranscriptSegment(
                speaker=UNKNOWN_SPEAKER,
                start=w_start,
                end=w_end,
                text=transcript_seg.text,
                alignment_confidence=0.0,
                overlap_duration=0.0,
                overlap_ratio=0.0,
                candidates=[],
            )

        # Build candidate list sorted by overlap duration descending
        candidates: List[SpeakerCandidateEvidence] = []
        for spk, dur in sorted(speaker_overlaps.items(), key=lambda x: x[1], reverse=True):
            ratio = min(1.0, dur / w_duration)
            candidates.append(
                SpeakerCandidateEvidence(
                    speaker=spk,
                    overlap_duration=round(dur, 3),
                    overlap_ratio=round(ratio, 3),
                )
            )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        # Check if best candidate has sufficient overlap
        if best.overlap_ratio < self.min_overlap_ratio:
            return AlignedTranscriptSegment(
                speaker=UNKNOWN_SPEAKER,
                start=w_start,
                end=w_end,
                text=transcript_seg.text,
                alignment_confidence=0.0,
                overlap_duration=best.overlap_duration,
                overlap_ratio=best.overlap_ratio,
                candidates=candidates,
            )

        # Single candidate case
        if second is None:
            # Deterministic confidence based on proportion of segment covered
            conf = round(min(1.0, best.overlap_ratio), 2)
            return AlignedTranscriptSegment(
                speaker=best.speaker,
                start=w_start,
                end=w_end,
                text=transcript_seg.text,
                alignment_confidence=conf,
                overlap_duration=best.overlap_duration,
                overlap_ratio=best.overlap_ratio,
                candidates=candidates,
            )

        # Multiple candidates case: check dominance
        is_clear_dominant = (
            best.overlap_ratio >= self.dominance_threshold
            and (best.overlap_ratio - second.overlap_ratio) >= self.dominance_ratio_margin
        )

        if is_clear_dominant:
            # Margin determines relative confidence
            margin = best.overlap_ratio - second.overlap_ratio
            conf = round(min(1.0, best.overlap_ratio * (margin + 0.5)), 2)
            return AlignedTranscriptSegment(
                speaker=best.speaker,
                start=w_start,
                end=w_end,
                text=transcript_seg.text,
                alignment_confidence=conf,
                overlap_duration=best.overlap_duration,
                overlap_ratio=best.overlap_ratio,
                candidates=candidates,
            )

        # Insufficient dominance / ambiguous overlap
        return AlignedTranscriptSegment(
            speaker=UNKNOWN_SPEAKER,
            start=w_start,
            end=w_end,
            text=transcript_seg.text,
            alignment_confidence=0.0,
            overlap_duration=best.overlap_duration,
            overlap_ratio=best.overlap_ratio,
            candidates=candidates,
        )

    def _merge_adjacent_aligned_segments(
        self,
        segments: List[AlignedTranscriptSegment],
        diarization_segments: List[SpeakerSegment],
    ) -> List[AlignedTranscriptSegment]:
        """Merge consecutive transcript segments if:

        1. Same non-UNKNOWN speaker
        2. Gap <= merge_gap_s
        3. No intervening diarization speaker activity during the gap
        """
        if not segments:
            return []

        merged: List[AlignedTranscriptSegment] = []
        curr = segments[0]

        for nxt in segments[1:]:
            gap = nxt.start - curr.end
            can_merge = (
                curr.speaker != UNKNOWN_SPEAKER
                and curr.speaker == nxt.speaker
                and 0.0 <= gap <= self.merge_gap_s
                and not self._has_intervening_speaker(
                    curr.end,
                    nxt.start,
                    target_speaker=curr.speaker,
                    diarization_segments=diarization_segments,
                )
            )

            if can_merge:
                # Merge text cleanly without duplicating words or losing spaces
                merged_text = f"{curr.text} {nxt.text}".strip()
                # Aggregate duration and weighted confidence
                merged_overlap_dur = round(curr.overlap_duration + nxt.overlap_duration, 3)
                total_dur = (nxt.end - curr.start)
                merged_overlap_ratio = round(min(1.0, merged_overlap_dur / max(1e-6, total_dur)), 3)
                merged_conf = round((curr.alignment_confidence + nxt.alignment_confidence) / 2.0, 2)

                curr = AlignedTranscriptSegment(
                    speaker=curr.speaker,
                    start=curr.start,
                    end=nxt.end,
                    text=merged_text,
                    alignment_confidence=merged_conf,
                    overlap_duration=merged_overlap_dur,
                    overlap_ratio=merged_overlap_ratio,
                    candidates=curr.candidates,
                )
            else:
                merged.append(curr)
                curr = nxt

        merged.append(curr)
        return merged

    def _has_intervening_speaker(
        self,
        gap_start: float,
        gap_end: float,
        target_speaker: str,
        diarization_segments: List[SpeakerSegment],
    ) -> bool:
        """Return True if any other speaker was active during the gap."""
        if gap_end <= gap_start:
            return False

        for d_seg in diarization_segments:
            # Overlaps the gap
            overlap_start = max(gap_start, d_seg.start)
            overlap_end = min(gap_end, d_seg.end)
            if (overlap_end - overlap_start) > 0.05:  # significant activity (> 50ms)
                if d_seg.speaker != target_speaker:
                    return True
        return False
