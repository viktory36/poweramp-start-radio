"""Complete-stream source decoding with a strict ffmpeg fallback."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from .server_config import (
    FFMPEG_SOURCE_DECODER_ID,
    TORCHAUDIO_SOURCE_DECODER_ID,
)

logger = logging.getLogger(__name__)


class SourceAudioDecodeError(RuntimeError):
    """The primary and fallback decoders both failed to produce complete PCM."""


def _validate_decoded_audio(waveform, sample_rate: int, decoder_id: str):
    if waveform.ndim != 2:
        raise SourceAudioDecodeError(
            f"{decoder_id} returned rank-{waveform.ndim} PCM; expected channels x frames"
        )
    if sample_rate <= 0 or waveform.shape[0] <= 0 or waveform.shape[-1] <= 0:
        raise SourceAudioDecodeError(f"{decoder_id} returned an empty or invalid audio span")
    waveform = waveform.to(dtype=torch.float32)
    if not torch.isfinite(waveform).all():
        raise SourceAudioDecodeError(f"{decoder_id} returned non-finite PCM")
    return waveform.contiguous(), int(sample_rate), int(waveform.shape[-1]), decoder_id


def _ffmpeg_source_audio(fpath: Path):
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise SourceAudioDecodeError(
            "ffmpeg fallback requires both ffmpeg and ffprobe on PATH"
        )

    probe = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "json",
            str(fpath),
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise SourceAudioDecodeError(f"ffprobe failed: {detail or 'no diagnostic'}")
    try:
        streams = json.loads(probe.stdout.decode("utf-8"))["streams"]
        if len(streams) != 1:
            raise ValueError(f"expected one first audio stream, found {len(streams)}")
        sample_rate = int(streams[0]["sample_rate"])
        channels = int(streams[0]["channels"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SourceAudioDecodeError(f"ffprobe returned invalid audio geometry: {error}") from error
    if sample_rate <= 0 or channels <= 0:
        raise SourceAudioDecodeError(
            f"ffprobe returned invalid audio geometry: {sample_rate} Hz, {channels} channels"
        )

    decoded = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v", "error",
            "-i", str(fpath),
            "-map", "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", "pcm_f32le",
            "-f", "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if decoded.returncode != 0:
        detail = decoded.stderr.decode("utf-8", errors="replace").strip()[-1000:]
        raise SourceAudioDecodeError(f"ffmpeg failed: {detail or 'no diagnostic'}")
    frame_bytes = channels * np.dtype("<f4").itemsize
    if not decoded.stdout or len(decoded.stdout) % frame_bytes != 0:
        raise SourceAudioDecodeError(
            "ffmpeg returned empty or non-frame-aligned native-rate PCM"
        )
    frames = len(decoded.stdout) // frame_bytes
    samples = np.frombuffer(decoded.stdout, dtype="<f4").reshape(frames, channels)
    waveform = torch.from_numpy(samples.T.copy())
    return _validate_decoded_audio(
        waveform,
        sample_rate,
        FFMPEG_SOURCE_DECODER_ID,
    )


def load_source_audio(fpath: Path):
    """Decode the complete first audio stream and retain its exact native span."""
    import torchaudio

    try:
        waveform, sample_rate = torchaudio.load(str(fpath))
        return _validate_decoded_audio(
            waveform,
            int(sample_rate),
            TORCHAUDIO_SOURCE_DECODER_ID,
        )
    except Exception as primary_error:
        logger.warning(
            "torchaudio could not decode %s; trying complete-stream ffmpeg fallback: %s",
            Path(fpath).name,
            primary_error,
        )
        try:
            return _ffmpeg_source_audio(fpath)
        except Exception as fallback_error:
            raise SourceAudioDecodeError(
                "audio decode failed with torchaudio "
                f"({primary_error}) and ffmpeg ({fallback_error})"
            ) from fallback_error
