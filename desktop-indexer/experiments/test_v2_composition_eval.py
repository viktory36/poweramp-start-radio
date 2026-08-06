from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_composition_eval as composition


def test_percentiles_are_ordered_and_tied_by_track_id() -> None:
    similarities = np.asarray([0.2, 0.9, 0.2, -0.1], dtype=np.float32)
    track_ids = np.asarray([20, 40, 10, 30], dtype=np.int64)
    result = composition.similarity_percentiles(similarities, track_ids)
    assert result.tolist() == [0.75, 1.0, 0.5, 0.25]


def test_signed_centroid_rejects_exact_contradiction() -> None:
    vector = np.asarray([1.0, 0.0], dtype=np.float32)
    anchors = [
        composition.Anchor("a", "a", vector, 0.5),
        composition.Anchor("a", "a", vector, -0.5),
    ]
    assert composition.normalized_signed_centroid(anchors) is None


def test_signed_centroid_is_invariant_to_positive_weight_scale() -> None:
    a = np.asarray([1.0, 0.0], dtype=np.float32)
    b = np.asarray([0.0, 1.0], dtype=np.float32)
    first = composition.normalized_signed_centroid(
        [
            composition.Anchor("a", "a", a, 0.6),
            composition.Anchor("b", "b", b, -0.4),
        ]
    )
    second = composition.normalized_signed_centroid(
        [
            composition.Anchor("a", "a", a, 6.0),
            composition.Anchor("b", "b", b, -4.0),
        ]
    )
    assert first is not None and second is not None
    np.testing.assert_allclose(first, second, atol=1e-7)


def test_either_rejects_negative_anchor() -> None:
    class TinyLibrary:
        count = 2

    anchor = composition.Anchor(
        "a", "a", np.asarray([1.0, 0.0], dtype=np.float32), -1.0
    )
    scores, error = composition.operator_scores(
        "either",
        TinyLibrary(),  # type: ignore[arg-type]
        [anchor],
        [np.asarray([0.5, 1.0])],
    )
    assert scores is None
    assert error == "Either does not accept negative anchors"
