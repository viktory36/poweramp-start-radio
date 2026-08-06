from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_capability_eval as capability


def test_gini_endpoints() -> None:
    assert capability.gini([1, 1, 1]) == 0.0
    assert abs(capability.gini([0, 0, 3]) - (2.0 / 3.0)) < 1e-12


def test_slerp_preserves_endpoints_and_unit_norm() -> None:
    start = np.asarray([1.0, 0.0], dtype=np.float32)
    end = np.asarray([0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(capability.slerp(start, end, 0.0), start, atol=1e-7)
    np.testing.assert_allclose(capability.slerp(start, end, 1.0), end, atol=1e-7)
    assert abs(np.linalg.norm(capability.slerp(start, end, 0.5)) - 1.0) < 1e-7


def test_kmedoids_is_repeat_deterministic() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.1],
            [0.0, 1.0],
            [0.1, 0.99],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    track_ids = np.asarray([40, 10, 30, 20], dtype=np.int64)
    first = capability.deterministic_kmedoids(embeddings, track_ids, 2)
    second = capability.deterministic_kmedoids(embeddings, track_ids, 2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert set(first[1].tolist()) == {0, 1}


def test_shortest_path_returns_none_across_components() -> None:
    graph = [{1: 1.0}, {0: 1.0}, {}]
    assert capability.shortest_path(graph, 0, 2) is None
    assert capability.shortest_path(graph, 0, 1) == [0, 1]


def test_shortest_path_does_not_cycle_across_equal_zero_cost_edges() -> None:
    graph = [
        {1: 0.0, 2: 0.0},
        {0: 0.0, 2: 0.0, 3: 1.0},
        {0: 0.0, 1: 0.0, 3: 1.0},
        {1: 1.0, 2: 1.0},
    ]
    assert capability.shortest_path(graph, 0, 3) == [0, 1, 3]
