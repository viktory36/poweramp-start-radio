from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_queue_eval as queue_eval
import v2_recording_identity_eval as identity_eval
import v2_seed_conditioning_eval as seed_eval


def library(embeddings: np.ndarray) -> queue_eval.Library:
    count = embeddings.shape[0]
    return queue_eval.Library(
        track_ids=np.arange(1, count + 1, dtype=np.int64),
        embeddings=embeddings.astype(np.float32),
        artists=tuple(f"artist-{index}" for index in range(count)),
        albums=tuple("album" for _ in range(count)),
        titles=tuple(f"track-{index}" for index in range(count)),
        durations_ms=np.full(count, 180_000, dtype=np.int64),
        clusters=np.arange(count, dtype=np.int32),
        sources=tuple("test" for _ in range(count)),
        metadata_keys=tuple(f"metadata-{index}" for index in range(count)),
        filename_keys=tuple(f"filename-{index}" for index in range(count)),
        file_paths=tuple(f"file-{index}" for index in range(count)),
    )


def test_seed_conditioned_dpp_uses_schur_complement_novelty() -> None:
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    almost_seed = np.array([0.995, 0.0998749, 0.0], dtype=np.float32)
    broader = np.array([0.8, 0.6, 0.0], dtype=np.float32)
    value = library(np.stack([seed, almost_seed, broader]))
    candidates = np.array([1, 2], dtype=np.int64)
    relevance = value.embeddings[candidates] @ seed
    gram = value.embeddings[candidates] @ value.embeddings[candidates].T

    canonical = seed_eval.select_dpp(
        value, 0, candidates, relevance, gram, 1, 1.0, False
    )
    conditioned = seed_eval.select_dpp(
        value, 0, candidates, relevance, gram, 1, 1.0, True
    )

    assert canonical.selected == (1,)
    assert conditioned.selected == (2,)


def test_seed_history_mmr_is_explicitly_distinct_from_empty_history() -> None:
    seed = np.array([1.0, 0.0], dtype=np.float32)
    close = np.array([0.99, 0.14106736], dtype=np.float32)
    farther = np.array([0.8, 0.6], dtype=np.float32)
    value = library(np.stack([seed, close, farther]))
    candidates = np.array([1, 2], dtype=np.int64)
    relevance = value.embeddings[candidates] @ seed
    gram = value.embeddings[candidates] @ value.embeddings[candidates].T

    canonical = seed_eval.select_mmr(
        value, 0, candidates, relevance, gram, 1, 0.4, False
    )
    seeded = seed_eval.select_mmr(
        value, 0, candidates, relevance, gram, 1, 0.4, True
    )

    assert canonical.selected == (1,)
    assert seeded.selected == (2,)


def test_cosine_duration_rule_is_only_a_broad_diagnostic() -> None:
    embeddings = np.array([[1.0, 0.0], [0.98, 0.1989975]], dtype=np.float32)
    value = library(embeddings)
    assert seed_eval.diagnostic_near_copy(value, 0, 1, 0.98, 0.95)
    value.durations_ms[1] = 300_000
    assert not seed_eval.diagnostic_near_copy(value, 0, 1, 0.98, 0.95)


def test_aligned_fingerprint_matcher_recovers_offset() -> None:
    prefix = [0x12345678 + index * 7919 for index in range(80)]
    shifted = [0xDEADBEEF] * 7 + prefix + [0xCAFEBABE] * 3
    result = identity_eval.aligned_fingerprint_comparison(prefix, shifted)
    assert result["score"] == 1.0
    assert result["alignment_offset_frames"] == -7


def test_aligned_fingerprint_matcher_accepts_two_bit_errors_only() -> None:
    base = [0x12345678 + index * 31 for index in range(40)]
    two_bits = [value ^ 0b11 for value in base]
    three_bits = [value ^ 0b111 for value in base]
    assert identity_eval.aligned_fingerprint_comparison(base, two_bits)["score"] == 1.0
    assert identity_eval.aligned_fingerprint_comparison(base, three_bits)["score"] < 1.0
