from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

import v2_queue_eval as queue_eval
import v2_text_eval as initial_eval
import v2_text_followup as followup


def tiny_library() -> queue_eval.Library:
    return queue_eval.Library(
        track_ids=np.asarray([9, 3, 7], dtype=np.int64),
        embeddings=np.eye(3, dtype=np.float32),
        artists=("Beyoncé", "BEYONCÉ", None),
        albums=("One", "Two", None),
        titles=("Song!", "song", "Unknown"),
        durations_ms=np.asarray([100, 100, 100], dtype=np.int64),
        clusters=np.asarray([-1, -1, -1], dtype=np.int32),
        sources=("desktop", "phone", "desktop"),
        metadata_keys=("a", "b", "c"),
        filename_keys=("a", "b", "c"),
        file_paths=("A.flac", "B.flac", "C.flac"),
    )


def test_canonical_text_normalizes_unicode_case_and_punctuation() -> None:
    assert followup.canonical_text("Beyoncé - Song!") == followup.canonical_text(
        "BEYONCÉ song"
    )


def test_rank_order_breaks_score_ties_by_track_id() -> None:
    order = initial_eval.rank_order(
        np.asarray([0.5, 0.5, 0.4], dtype=np.float32),
        np.asarray([9, 3, 7], dtype=np.int64),
    )
    assert order.tolist() == [1, 0, 2]


def test_crowding_detects_normalized_artist_title_duplicates() -> None:
    metrics = followup.crowding_metrics(
        tiny_library(), np.asarray([0, 1, 2], dtype=np.int64), 3
    )
    assert metrics["normalized_artist_title_duplicate_excess"] == 1
    assert metrics["metadata_key_duplicate_excess"] == 0
    assert metrics["file_path_duplicate_excess"] == 0


def test_overlap_reports_fraction_and_jaccard() -> None:
    result = followup.overlap([1, 2, 3], [2, 3, 4])
    assert result == {
        "intersection": 2,
        "overlap_fraction": 2 / 3,
        "jaccard": 0.5,
    }
