from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_active_composition_eval as active


def test_empirical_percentiles_preserve_equal_score_ties() -> None:
    values = np.asarray([0.2, 0.8, 0.8, -0.1], dtype=np.float32)
    assert active.empirical_percentiles(values).tolist() == [0.5, 1.0, 1.0, 0.25]


def test_all_of_is_weighted_geometric_mean() -> None:
    first = np.asarray([1.0, 0.25, 0.5, 0.75], dtype=np.float32)
    second = np.asarray([0.25, 1.0, 0.75, 0.5], dtype=np.float32)
    result = active.all_of_scores([first, second], [0.5, 0.5])
    np.testing.assert_allclose(result[:2], [0.5, 0.5], atol=1e-7)


def test_either_uses_weighted_prefix_fairness_and_refills() -> None:
    track_ids = np.arange(1, 10, dtype=np.int64)
    branches = [
        np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]),
        np.asarray([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
    ]
    selected, winners = active.rank_either(
        track_ids, branches, [0.5, 0.5], {1}, count=6
    )
    assert track_ids[selected].tolist() == [2, 9, 3, 8, 4, 7]
    assert winners.tolist() == [0, 1, 0, 1, 0, 1]


def test_signed_centroid_rejects_exact_cancellation() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    anchors = [
        active.Anchor("a", "a", vector, 0.5),
        active.Anchor("a", "a", vector, -0.5),
    ]
    assert active.normalized_signed_centroid(anchors) is None


def test_variance_calibration_scales_anchor_vectors_before_normalizing() -> None:
    anchors = [
        active.Anchor("a", "a", np.asarray([1.0, 0.0], dtype=np.float32), 0.5),
        active.Anchor("b", "b", np.asarray([0.0, 1.0], dtype=np.float32), 0.5),
    ]
    result = active.normalized_variance_calibrated_centroid(anchors, [0.5, 0.25])
    assert result is not None
    np.testing.assert_allclose(
        result,
        np.asarray([1.0, 2.0]) / np.sqrt(5.0),
        atol=1e-7,
    )


def test_strict_rank_uses_geo_mean_then_id_as_ties() -> None:
    ids = np.asarray([30, 10, 20], dtype=np.int64)
    strict = np.asarray([0.8, 0.8, 0.7], dtype=np.float32)
    geo = np.asarray([0.85, 0.90, 0.99], dtype=np.float32)
    order = active.rank_scalar(ids, strict, set(), 3, tie_scores=geo)
    assert ids[order].tolist() == [10, 30, 20]


def test_diagnostic_power_means_and_rrf_have_expected_balance() -> None:
    first = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    second = np.asarray([0.25, 0.5, 1.0], dtype=np.float32)
    arithmetic = active.diagnostic_aggregate_scores(
        "arithmetic_mean", [first, second], [0.5, 0.5]
    )
    harmonic = active.diagnostic_aggregate_scores(
        "harmonic_mean", [first, second], [0.5, 0.5]
    )
    rrf = active.diagnostic_aggregate_scores("rrf_k60", [first, second], [0.5, 0.5])
    assert arithmetic.tolist() == [0.625, 0.5, 0.625]
    np.testing.assert_allclose(harmonic, [0.4, 0.5, 0.4], atol=1e-7)
    assert rrf[0] > rrf[1]
    assert rrf[2] > rrf[1]
    assert rrf[0] == rrf[2]


def test_two_dimensional_pareto_depths_are_exact() -> None:
    first = np.asarray([1.0, 0.8, 0.7, 0.5], dtype=np.float32)
    second = np.asarray([0.5, 0.9, 0.4, 0.3], dtype=np.float32)
    # Rows 0 and 1 trade off on front 1; row 2 is dominated by both, then row 3.
    assert active.pareto_depths_2d([first, second]).tolist() == [1, 1, 2, 3]


def test_three_dimensional_first_front_is_not_a_total_order() -> None:
    values = np.asarray(
        [
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.4, 0.4, 0.4],
        ],
        dtype=np.float32,
    )
    assert active.nondominated_mask_3d(values).tolist() == [True, True, True, False]
