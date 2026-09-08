"""Audio processor implementing validation, persistence, FFprobe metadata extraction, and FFmpeg standardization."""
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from typing import BinaryIO, Optional, Union

from backend.app.core.config import Settings, get_settings
from backend.app.services.meeting_pipeline.audio.exceptions import (
    AudioConversionError,
    AudioValidationError,
)
from backend.app.services.meeting_pipeline.audio.models import (
    AudioMetadata,
    AudioValidationResult,
    ProcessedAudio,
    QualityReport,
    VADResult,
)


class AudioProcessor:
    """Handles complete audio lifecycle:
    - Format and size validation
    - Secure storage persistence
    - Metadata extraction via ffprobe
    - 7-Stage Preprocessing Pipeline:
      1. Decode (any input audio via pydub / ffmpeg)
      2. Standardize (Mono, 16 kHz, float32 headroom)
      3. Quality Check (Loudness, clipping, noise floor, silence)
      4. Conditional Enhancement (Light denoise if noisy, gain if too quiet)
      5. High-pass filter (~70-80 Hz, rumble & DC offset removal)
      6. VAD (WebRTC Voice Activity Detection stats, full duration, no chunking)
      7. Final WAV (PCM 16-bit, 16 kHz mono WAV)
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.raw_dir = self.settings.raw_storage_path
        self.processed_dir = self.settings.processed_storage_path
        self._ensure_storage_dirs()

    def _ensure_storage_dirs(self) -> None:
        """Ensure raw and processed storage directories exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_metadata(self, file_path: Union[str, Path]) -> AudioMetadata:
        """Extract audio stream metadata using ffprobe."""
        path = Path(file_path)
        if not path.exists():
            raise AudioValidationError(f"Audio file does not exist: {path}")

        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            probe_data = json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            raise AudioConversionError(f"ffprobe failed to read file {path}: {exc.stderr}") from exc
        except json.JSONDecodeError as exc:
            raise AudioConversionError(f"Failed to parse ffprobe output for {path}") from exc

        audio_stream = next(
            (s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"),
            None,
        )

        fmt_data = probe_data.get("format", {})
        duration_sec = 0.0
        if audio_stream and "duration" in audio_stream and audio_stream["duration"] is not None:
            duration_sec = float(audio_stream["duration"])
        elif "duration" in fmt_data and fmt_data["duration"] is not None:
            duration_sec = float(fmt_data["duration"])

        sample_rate = int(audio_stream.get("sample_rate", self.settings.target_sample_rate)) if audio_stream else self.settings.target_sample_rate
        channels = int(audio_stream.get("channels", self.settings.target_channels)) if audio_stream else self.settings.target_channels
        size_bytes = int(fmt_data.get("size", path.stat().st_size))

        return AudioMetadata(
            file_path=path,
            file_name=path.name,
            file_size_bytes=size_bytes,
            duration_seconds=max(0.0, duration_sec),
            sample_rate=sample_rate,
            channels=channels,
            format=path.suffix.lower().lstrip("."),
        )

    def validate_input(
        self,
        file_path: Union[str, Path],
        file_size_bytes: Optional[int] = None,
    ) -> AudioValidationResult:
        """Validate input file existence, extension, and size limits."""
        path = Path(file_path)
        errors = []

        if not path.exists():
            return AudioValidationResult(
                is_valid=False,
                errors=[f"Audio file does not exist: {path}"],
                metadata=None,
            )

        suffix = path.suffix.lower().lstrip(".")
        if suffix not in self.settings.allowed_audio_extensions:
            errors.append(
                f"Unsupported audio format '{suffix}'. Allowed formats: {self.settings.allowed_audio_extensions}"
            )

        size = file_size_bytes if file_size_bytes is not None else path.stat().st_size
        max_bytes = self.settings.max_audio_size_mb * 1024 * 1024
        if size > max_bytes:
            errors.append(
                f"File size ({size / (1024 * 1024):.2f}MB) exceeds limit of {self.settings.max_audio_size_mb}MB"
            )
        elif size == 0:
            errors.append("File is empty (0 bytes)")

        if errors:
            return AudioValidationResult(is_valid=False, errors=errors, metadata=None)

        try:
            metadata = self.extract_metadata(path)
        except Exception:
            # Fallback if ffprobe fails on partial/mock files during early stage
            metadata = AudioMetadata(
                file_path=path,
                file_name=path.name,
                file_size_bytes=size,
                duration_seconds=0.0,
                sample_rate=self.settings.target_sample_rate,
                channels=self.settings.target_channels,
                format=suffix,
            )

        return AudioValidationResult(is_valid=True, errors=[], metadata=metadata)

    def save_original(
        self,
        file_content: Union[bytes, BinaryIO],
        original_filename: str,
        job_id: Optional[str] = None,
    ) -> Path:
        """Persist original incoming audio file into raw storage directory."""
        job_id = job_id or uuid.uuid4().hex
        ext = Path(original_filename).suffix.lower()
        destination = self.raw_dir / f"{job_id}{ext}"

        if isinstance(file_content, bytes):
            with open(destination, "wb") as f:
                f.write(file_content)
        else:
            with open(destination, "wb") as f:
                chunk = file_content.read(1024 * 1024)
                while chunk:
                    f.write(chunk)
                    chunk = file_content.read(1024 * 1024)

        return destination

    # --------------------------------------------------------------------------
    # 7-Stage Preprocessing Pipeline Methods
    # --------------------------------------------------------------------------
    @staticmethod
    def decode_audio(path: Path):
        """1. DECODE any common audio format using pydub/ffmpeg to float32 samples in [-1, 1]."""
        from pydub import AudioSegment
        import numpy as np

        seg = AudioSegment.from_file(str(path))
        sr = seg.frame_rate
        channels = seg.channels

        samples = np.array(seg.get_array_of_samples())
        if channels > 1:
            samples = samples.reshape((-1, channels))

        max_val = float(1 << (8 * seg.sample_width - 1))
        audio = samples.astype(np.float32) / max_val
        return audio, sr

    @staticmethod
    def standardize_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> tuple:
        """2. STANDARDIZE: Convert to Mono and resample to 16kHz."""
        import numpy as np
        from math import gcd
        from scipy.signal import resample_poly

        # Mono conversion
        if audio.ndim > 1:
            mono = audio.mean(axis=1).astype(np.float32)
        else:
            mono = audio.astype(np.float32)

        # Resample
        if orig_sr == target_sr:
            resampled = mono
        else:
            g = gcd(orig_sr, target_sr)
            up, down = target_sr // g, orig_sr // g
            resampled = resample_poly(mono, up, down).astype(np.float32)

        resampled = np.clip(resampled, -1.0, 1.0)
        return resampled, target_sr

    @staticmethod
    def _dbfs(x: np.ndarray) -> float:
        import numpy as np
        rms = np.sqrt(np.mean(np.square(x))) if x.size else 0.0
        if rms <= 1e-9:
            return -120.0
        return float(20 * np.log10(rms))

    def quality_check(self, audio: np.ndarray, sr: int, frame_ms: int = 30) -> QualityReport:
        """3. QUALITY CHECK: Loudness, clipping, noise floor, and silence ratio."""
        import numpy as np

        loudness = self._dbfs(audio)
        clipping_ratio = float(np.mean(np.abs(audio) >= self.settings.audio_clip_threshold))
        is_clipped = clipping_ratio > 0.001

        frame_len = int(sr * frame_ms / 1000)
        n_frames = max(1, len(audio) // frame_len)
        frame_dbfs = []
        for i in range(n_frames):
            frame = audio[i * frame_len : (i + 1) * frame_len]
            frame_dbfs.append(self._dbfs(frame))
        frame_dbfs_arr = np.array(frame_dbfs)

        silent_frames = frame_dbfs_arr < self.settings.audio_silence_threshold_dbfs
        silence_ratio = float(np.mean(silent_frames)) if len(silent_frames) else 0.0
        is_mostly_silent = silence_ratio > 0.85

        noise_floor = -120.0
        if len(frame_dbfs_arr):
            sorted_frames = np.sort(frame_dbfs_arr)
            quiet_slice = sorted_frames[: max(1, len(sorted_frames) // 5)]
            noise_floor = float(np.median(quiet_slice))

        is_noisy = noise_floor > self.settings.audio_noise_floor_threshold_dbfs
        is_too_quiet = loudness < self.settings.audio_low_loudness_dbfs

        notes = []
        if is_clipped:
            notes.append(f"Clipping detected ({clipping_ratio * 100:.2f}% samples)")
        if is_too_quiet:
            notes.append(f"Audio is quiet ({loudness:.1f} dBFS)")
        if is_noisy:
            notes.append(f"High noise floor ({noise_floor:.1f} dBFS)")
        if is_mostly_silent:
            notes.append(f"Mostly silence ({silence_ratio * 100:.1f}% frames)")

        return QualityReport(
            loudness_dbfs=round(loudness, 2),
            clipping_ratio=round(clipping_ratio, 4),
            noise_floor_dbfs=round(noise_floor, 2),
            silence_ratio=round(silence_ratio, 4),
            is_clipped=is_clipped,
            is_too_quiet=is_too_quiet,
            is_noisy=is_noisy,
            is_mostly_silent=is_mostly_silent,
            notes=notes,
        )

    def conditional_enhancement(
        self,
        audio: np.ndarray,
        sr: int,
        report: QualityReport,
        enable_denoise: bool = True,
    ) -> np.ndarray:
        """4. CONDITIONAL ENHANCEMENT: Light denoise (if noisy) + gain (if quiet)."""
        import numpy as np
        out = audio

        # Light denoise
        if report.is_noisy and enable_denoise:
            try:
                import noisereduce as nr
                out = nr.reduce_noise(y=out, sr=sr, prop_decrease=0.6, stationary=False).astype(np.float32)
            except Exception:
                pass

        # Gain normalization if too quiet
        if report.is_too_quiet:
            gain_db = self.settings.audio_target_loudness_dbfs - report.loudness_dbfs
            gain_db = float(np.clip(gain_db, -12.0, 12.0))
            gain_lin = 10.0 ** (gain_db / 20.0)
            out = np.clip(out * gain_lin, -1.0, 1.0).astype(np.float32)

        return out

    def high_pass_filter(self, audio: np.ndarray, sr: int, cutoff: Optional[float] = None, order: int = 4) -> np.ndarray:
        """5. HIGH-PASS FILTER (~70-80 Hz): Removes rumble & DC offset."""
        import numpy as np
        from scipy.signal import butter, sosfiltfilt

        cutoff_freq = cutoff or self.settings.audio_highpass_cutoff
        nyq = sr / 2.0
        normalized_cutoff = min(0.99, cutoff_freq / nyq)
        sos = butter(order, normalized_cutoff, btype="highpass", output="sos")
        filtered = sosfiltfilt(sos, audio)
        return filtered.astype(np.float32)

    @staticmethod
    def detect_speech(audio: np.ndarray, sr: int, frame_ms: int = 30, aggressiveness: int = 2) -> VADResult:
        """6. VAD: WebRTC Voice Activity Detection across full clip without chunking."""
        import numpy as np
        try:
            import webrtcvad
            vad = webrtcvad.Vad(aggressiveness)
        except Exception:
            return VADResult(speech_ratio=None, has_speech=None, segments=[])

        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        frame_len = int(sr * frame_ms / 1000)
        bytes_per_frame = frame_len * 2
        raw = pcm16.tobytes()
        n_frames = len(raw) // bytes_per_frame

        speech_flags = []
        segments = []
        in_speech = False
        seg_start = 0.0

        for i in range(n_frames):
            frame_bytes = raw[i * bytes_per_frame : (i + 1) * bytes_per_frame]
            is_speech = vad.is_speech(frame_bytes, sr)
            speech_flags.append(is_speech)

            t = i * frame_ms / 1000.0
            if is_speech and not in_speech:
                in_speech = True
                seg_start = t
            elif not is_speech and in_speech:
                in_speech = False
                segments.append((round(seg_start, 3), round(t, 3)))

        if in_speech:
            segments.append((round(seg_start, 3), round(n_frames * frame_ms / 1000.0, 3)))

        speech_ratio = float(np.mean(speech_flags)) if speech_flags else 0.0

        return VADResult(
            speech_ratio=round(speech_ratio, 4),
            has_speech=speech_ratio > 0.02,
            segments=segments,
        )

    @staticmethod
    def write_final_wav(audio: np.ndarray, sr: int, out_path: Path) -> None:
        """7. FINAL WAV: Write 16-bit PCM WAV (full duration, no chunking)."""
        import numpy as np
        import soundfile as sf

        clipped = np.clip(audio, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        sf.write(str(out_path), pcm16, sr, subtype="PCM_16")

    def preprocess(
        self,
        input_path: Path,
        job_id: Optional[str] = None,
        enable_denoise: Optional[bool] = None,
    ) -> ProcessedAudio:
        """Execute the 7-Stage Preprocessing Pipeline:
        RAW AUDIO -> Decode -> Standardize (16kHz Mono) -> Quality Check ->
        Conditional Enhancement (Denoise/Gain) -> High-pass Filter (75Hz) ->
        VAD (detect speech) -> Final WAV (Full duration, no chunking).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise AudioValidationError(f"Cannot preprocess non-existent audio: {input_path}")

        job_id = job_id or uuid.uuid4().hex
        output_filename = f"{job_id}_standardized.wav"
        output_path = self.processed_dir / output_filename

        should_denoise = enable_denoise if enable_denoise is not None else self.settings.audio_enable_denoise

        try:
            # 1. Decode
            audio, sr = self.decode_audio(input_path)

            # 2. Standardize to 16kHz mono float32
            audio, target_sr = self.standardize_audio(audio, sr, target_sr=self.settings.target_sample_rate)

            # 3. Quality Check
            report = self.quality_check(audio, target_sr)

            # 4. Conditional Enhancement (Light denoise & gain)
            audio = self.conditional_enhancement(audio, target_sr, report, enable_denoise=should_denoise)

            # 5. High-pass filter (~75Hz)
            audio = self.high_pass_filter(audio, target_sr, cutoff=self.settings.audio_highpass_cutoff)

            # 6. VAD (Detect speech stats, full duration kept)
            vad_res = self.detect_speech(audio, target_sr)

            # 7. Final WAV write
            self.write_final_wav(audio, target_sr, output_path)

            duration_seconds = max(0.0, float(len(audio)) / float(target_sr))

            return ProcessedAudio(
                job_id=job_id,
                processed_path=output_path,
                original_path=input_path,
                duration_seconds=round(duration_seconds, 3),
                sample_rate=target_sr,
                channels=self.settings.target_channels,
                format="wav",
                quality_report=report,
                vad_result=vad_res,
            )
        except Exception as exc:
            # Fallback to FFmpeg direct conversion if numpy pipeline hits unexpected container issue
            try:
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-vn",
                    "-ac",
                    str(self.settings.target_channels),
                    "-ar",
                    str(self.settings.target_sample_rate),
                    "-c:a",
                    "pcm_s16le",
                    str(output_path),
                ]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                meta = self.extract_metadata(output_path)
                return ProcessedAudio(
                    job_id=job_id,
                    processed_path=output_path,
                    original_path=input_path,
                    duration_seconds=meta.duration_seconds,
                    sample_rate=meta.sample_rate,
                    channels=meta.channels,
                    format="wav",
                )
            except Exception as ffmpeg_exc:
                raise AudioConversionError(
                    f"Audio preprocessing pipeline failed for {input_path}: {exc} (ffmpeg fallback failed: {ffmpeg_exc})"
                ) from exc

