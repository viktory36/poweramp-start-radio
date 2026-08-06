from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_focused_recommendation_eval as focused


def test_stable_order_uses_track_id_for_ties() -> None:
    scores = np.asarray([0.2, 0.8, 0.8, -0.1], dtype=np.float32)
    track_ids = np.asarray([8, 9, 3, 2], dtype=np.int64)
    assert focused.stable_order(scores, track_ids).tolist() == [2, 1, 0, 3]


def test_tie_aware_percentile_is_monotonic_and_equal_for_ties() -> None:
    scores = np.asarray([4.0, 1.0, 1.0, 3.0], dtype=np.float32)
    percentiles = focused.tie_aware_upper_percentile(scores)
    assert percentiles.tolist() == [1.0, 0.5, 0.5, 0.75]


def test_rank_positions_inverts_order() -> None:
    order = np.asarray([2, 0, 3, 1], dtype=np.int64)
    assert focused.rank_positions(order).tolist() == [2, 4, 1, 3]


def test_all_of_rewards_joint_satisfaction() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2**-0.5, 2**-0.5],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    library = focused.Library(
        track_ids=np.arange(1, 5, dtype=np.int64),
        embeddings=embeddings,
        artists=("",) * 4,
        albums=("",) * 4,
        titles=("",) * 4,
        metadata_keys=("a", "b", "c", "d"),
        filename_keys=("a", "b", "c", "d"),
        file_paths=("a", "b", "c", "d"),
        durations_ms=np.ones(4, dtype=np.int64),
        index_by_track_id={i + 1: i for i in range(4)},
    )
    objective, per_anchor = focused.all_of_objective(
        library,
        [embeddings[0], embeddings[1]],
    )
    assert int(np.argmax(objective)) == 2
    assert per_anchor.shape == (2, 4)


def test_constrained_prefix_applies_spacing_deterministically() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    library = focused.Library(
        track_ids=np.arange(1, 5, dtype=np.int64),
        embeddings=embeddings,
        artists=("same", "same", "other", "same"),
        albums=("",) * 4,
        titles=("",) * 4,
        metadata_keys=("a", "b", "c", "d"),
        filename_keys=("a", "b", "c", "d"),
        file_paths=("a", "b", "c", "d"),
        durations_ms=np.ones(4, dtype=np.int64),
        index_by_track_id={i + 1: i for i in range(4)},
    )
    assert focused.constrained_prefix(
        library,
        [0, 1, 2, 3],
        count=3,
        excluded=set(),
    ) == [0, 2]
