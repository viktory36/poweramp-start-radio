from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from v2_audio_embedding_parity import (
    EMBEDDING_DIM,
    POLICY_ID,
    planned_work,
    sha256,
    torchaudio_target_length,
    validate_fixture_manifest,
)


def test_torchaudio_target_length_matches_pinned_exact_sample_examples() -> None:
    assert torchaudio_target_length(972_272, 44_100) == 529_128
    assert torchaudio_target_length(455_700, 44_100) == 248_000
    assert torchaudio_target_length(2_646_519, 48_000) == 1_323_260
    assert torchaudio_target_length(123_456, 24_000) == 123_456


def test_work_policy_applies_one_second_tail_floor_and_bookends() -> None:
    assert planned_work(23_999) == (0, 0, 23_999, False)
    assert planned_work(24_000) == (1, 1, 24_000, True)
    assert planned_work(119_999) == (1, 1, 119_999, True)
    assert planned_work(120_000) == (1, 1, 0, False)
    assert planned_work(15_120_000) == (126, 1, 0, False)
    assert planned_work(15_240_000) == (127, 2, 0, False)


def test_fixture_validation_binds_source_vector_hashes_and_work(tmp_path: Path) -> None:
    source = tmp_path / "source" / "track.flac"
    source.parent.mkdir()
    source.write_bytes(b"lossless-source-bytes")
    expected = tmp_path / "expected" / "track.f32le"
    expected.parent.mkdir()
    expected.write_bytes(np.zeros(EMBEDDING_DIM, dtype="<f4").tobytes())
    canonical_samples = 248_000
    windows, segments, tail, tail_included = planned_work(canonical_samples)
    manifest = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "tracks": [
            {
                "name": "track",
                "source_file": "source/track.flac",
                "source_sha256": sha256(source),
                "embedding_file": "expected/track.f32le",
                "embedding_sha256": sha256(expected),
                "canonical_sample_count_24k": canonical_samples,
                "mert_windows": windows,
                "clamp_segments": segments,
                "tail_samples_24k": tail,
                "tail_included": tail_included,
            },
        ],
    }

    validate_fixture_manifest(manifest, tmp_path)
    expected.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="embedding hash mismatch"):
        validate_fixture_manifest(manifest, tmp_path)
