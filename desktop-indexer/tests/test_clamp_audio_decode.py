from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torchaudio

from poweramp_indexer import source_audio
from poweramp_indexer.server_config import (
    FFMPEG_SOURCE_DECODER_ID,
    TORCHAUDIO_SOURCE_DECODER_ID,
)


def test_torchaudio_is_primary_and_reports_exact_native_span(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "Café 夜.mp3"
    source.write_bytes(b"fixture")
    expected = torch.tensor([[0.25, -0.5, 0.75]], dtype=torch.float32)
    monkeypatch.setattr(torchaudio, "load", lambda _path: (expected, 44_100))

    def unexpected_fallback(_path):
        raise AssertionError("ffmpeg fallback must not run after a valid primary decode")

    monkeypatch.setattr(source_audio, "_ffmpeg_source_audio", unexpected_fallback)

    waveform, sample_rate, sample_count, decoder_id = (
        source_audio.load_source_audio(source)
    )

    assert torch.equal(waveform, expected)
    assert sample_rate == 44_100
    assert sample_count == 3
    assert decoder_id == TORCHAUDIO_SOURCE_DECODER_ID


def test_ffmpeg_fallback_decodes_complete_first_stream_at_native_geometry(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "nearly valid 夜.mp3"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        torchaudio,
        "load",
        lambda _path: (_ for _ in ()).throw(RuntimeError("torchaudio rejected headers")),
    )
    monkeypatch.setattr(
        source_audio.shutil,
        "which",
        lambda executable: f"/tools/{executable}",
    )
    interleaved = np.array(
        [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]],
        dtype="<f4",
    )
    commands: list[list[str]] = []

    def run(command, **kwargs):
        assert kwargs == {
            "capture_output": True,
            "check": False,
        }
        commands.append(command)
        if command[0].endswith("ffprobe"):
            payload = {"streams": [{"sample_rate": "48000", "channels": 2}]}
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(payload).encode("utf-8"),
                stderr=b"",
            )
        return SimpleNamespace(returncode=0, stdout=interleaved.tobytes(), stderr=b"")

    monkeypatch.setattr(source_audio.subprocess, "run", run)

    waveform, sample_rate, sample_count, decoder_id = (
        source_audio.load_source_audio(source)
    )

    assert sample_rate == 48_000
    assert sample_count == 3
    assert decoder_id == FFMPEG_SOURCE_DECODER_ID
    assert np.allclose(waveform.numpy(), interleaved.T)
    assert commands[0][-1] == str(source)
    decode = commands[1]
    assert decode[decode.index("-map") + 1] == "0:a:0"
    assert decode[decode.index("-ar") + 1] == "48000"
    assert decode[decode.index("-ac") + 1] == "2"
    assert "-ss" not in decode
    assert "-t" not in decode
    assert decode[-1] == "pipe:1"


def test_ffmpeg_fallback_rejects_partial_frames(tmp_path: Path, monkeypatch):
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        torchaudio,
        "load",
        lambda _path: (_ for _ in ()).throw(RuntimeError("primary failed")),
    )
    monkeypatch.setattr(
        source_audio.shutil,
        "which",
        lambda executable: f"/tools/{executable}",
    )
    responses = iter(
        (
            SimpleNamespace(
                returncode=0,
                stdout=b'{"streams":[{"sample_rate":"44100","channels":2}]}',
                stderr=b"",
            ),
            SimpleNamespace(returncode=0, stdout=b"\x00" * 7, stderr=b""),
        )
    )
    monkeypatch.setattr(
        source_audio.subprocess,
        "run",
        lambda _command, **_kwargs: next(responses),
    )

    with pytest.raises(
        source_audio.SourceAudioDecodeError,
        match="non-frame-aligned",
    ):
        source_audio.load_source_audio(source)
