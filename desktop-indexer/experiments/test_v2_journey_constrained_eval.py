from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_journey_constrained_eval as journey


def synthetic_library(
    embeddings: list[list[float]], artists: list[str | None]
) -> journey.JourneyLibrary:
    values = np.asarray(embeddings, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    count = len(embeddings)
    return journey.JourneyLibrary(
        track_ids=np.arange(100, 100 + count, dtype=np.int64),
        embeddings=values,
        artists=tuple(artists),
        albums=(None,) * count,
        titles=tuple(f"track {index}" for index in range(count)),
        durations_ms=np.zeros(count, dtype=np.int64),
        file_paths=("",) * count,
    )


def test_slerp_and_corridor_geometry_match_the_ideal_arc() -> None:
    start = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    destination = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    fractions = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    path = np.stack(
        [journey.slerp(start, destination, float(value)) for value in fractions]
    )
    geometry = journey.corridor_geometry(path, start, destination, fractions)

    np.testing.assert_allclose(geometry["route_progress"], fractions, atol=1e-6)
    np.testing.assert_allclose(geometry["cross_track"], 0.0, atol=1e-3)
    np.testing.assert_allclose(geometry["target_errors"], 0.0, atol=1e-3)


def test_artist_spacing_and_destination_reservation_are_explicit_state() -> None:
    library = synthetic_library(
        [[1, 0], [0.98, 0.2], [0.8, 0.6], [0, 1]],
        ["A", "B", "A", "B"],
    )
    assert journey.can_append_artist(2, [0], library) is False
    assert journey.can_append_artist(2, [0, 1], library) is False
    assert journey.reserve_destination_artist(1, 2, 4, 3, library) is False
    assert journey.reserve_destination_artist(2, 2, 4, 3, library) is True


def test_path_metrics_joint_objective_matches_declared_formula() -> None:
    root_half = math.sqrt(0.5)
    library = synthetic_library(
        [[1, 0, 0], [root_half, root_half, 0], [0, 1, 0]],
        ["A", "B", "C"],
    )
    metrics = journey.path_metrics([0, 1, 2], library, 0, 2, 3)
    edge_angle = math.pi / 4.0
    expected = (
        2.0 * journey.EDGE_WEIGHT * edge_angle * edge_angle
        + journey.MAX_EDGE_WEIGHT * edge_angle * edge_angle
    )
    assert math.isclose(
        float(metrics["joint_objective_cost"]), expected, rel_tol=1e-6, abs_tol=1e-6
    )
    assert metrics["destination_broad_violations"] == 0
    assert metrics["outlier_count_over_35_degrees"] == 0


def test_shortest_path_does_not_cycle_on_equal_zero_cost_edges() -> None:
    graph = [
        {1: 0.0, 2: 0.0},
        {0: 0.0, 2: 0.0, 3: 1.0},
        {0: 0.0, 1: 0.0, 3: 1.0},
        {1: 1.0, 2: 1.0},
    ]
    assert journey.shortest_path(graph, 0, 3) == [0, 1, 3]
