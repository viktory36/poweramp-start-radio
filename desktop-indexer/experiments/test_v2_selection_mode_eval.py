from __future__ import annotations

import numpy as np

import v2_queue_eval as queue_eval
import v2_selection_mode_eval as selection_eval


def make_library(embeddings: np.ndarray) -> queue_eval.Library:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    count = embeddings.shape[0]
    return queue_eval.Library(
        track_ids=np.arange(1, count + 1, dtype=np.int64),
        embeddings=embeddings,
        artists=tuple(f"artist-{index}" for index in range(count)),
        albums=tuple(None for _ in range(count)),
        titles=tuple(f"track-{index}" for index in range(count)),
        durations_ms=np.full(count, 180_000, dtype=np.int64),
        clusters=np.full(count, -1, dtype=np.int32),
        sources=tuple("test" for _ in range(count)),
        metadata_keys=tuple(f"metadata-{index}" for index in range(count)),
        filename_keys=tuple(f"filename-{index}" for index in range(count)),
        file_paths=tuple(f"/{index}.flac" for index in range(count)),
    )


def test_absolute_half_life_is_prefix_invariant_but_current_decay_is_not() -> None:
    base = 0.8440951
    current_10 = [selection_eval.current_exponential_anchor(base, step, 10) for step in range(10)]
    current_30 = [selection_eval.current_exponential_anchor(base, step, 30) for step in range(30)]
    fixed_10 = [selection_eval.fixed_half_life_anchor(base, step, 7.0) for step in range(10)]
    fixed_30 = [selection_eval.fixed_half_life_anchor(base, step, 7.0) for step in range(30)]

    assert current_10[1:] != current_30[1:10]
    assert fixed_10 == fixed_30[:10]
    assert np.isclose(current_30[-1], base * np.exp(-3.0))


def test_dpp_quality_exponent_is_a_real_deterministic_control() -> None:
    library = make_library(
        np.array(
            [
                [1.0, 0.0],
                [0.5, 0.8660254],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
    )
    candidates = np.array([0, 1, 2], dtype=np.int64)
    relevance = np.array([0.9, 0.85, 0.2], dtype=np.float32)

    diverse, _ = selection_eval.select_dpp_exponent(
        library, candidates, relevance, count=2, quality_exponent=0.0
    )
    focused, _ = selection_eval.select_dpp_exponent(
        library, candidates, relevance, count=2, quality_exponent=4.0
    )

    assert diverse == [0, 2]
    assert focused == [0, 1]
    assert selection_eval.select_dpp_exponent(
        library, candidates, relevance, count=2, quality_exponent=4.0
    )[0] == focused


def test_dpp_float64_reference_matches_float32_on_controlled_case() -> None:
    library = make_library(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.7, 0.0, 0.7141428],
                [0.3, 0.7, 0.6480741],
            ]
        )
    )
    candidates = np.arange(4, dtype=np.int64)
    relevance = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    float32 = selection_eval.select_dpp_exponent(
        library, candidates, relevance, count=3, quality_exponent=1.0
    )
    float64 = selection_eval.select_dpp_float64_reference(
        library, candidates, relevance, count=3, quality_exponent=1.0
    )
    assert float32 == float64


def test_facility_coverage_is_complete_deterministic_and_seed_fixed() -> None:
    library = make_library(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.31, 0.0],
                [0.70, 0.00, 0.71],
                [0.70, 0.71, 0.00],
                [0.58, 0.58, 0.58],
            ]
        )
    )
    candidates = np.arange(5, dtype=np.int64)
    first = selection_eval.select_facility_coverage(library, candidates, count=4)
    second = selection_eval.select_facility_coverage(library, candidates, count=4)

    assert first == second
    assert first[0][0] == 0
    assert len(first[0]) == 4
    assert len(set(first[0])) == 4


def test_rank_map_breaks_similarity_ties_by_track_id() -> None:
    similarities = np.array([0.5, 0.7, 0.7, 0.2], dtype=np.float32)
    track_ids = np.array([40, 30, 20, 10], dtype=np.int64)
    ranks = selection_eval.rank_map(similarities, track_ids)
    assert ranks.tolist() == [3, 2, 1, 4]


def test_exact_graph_distribution_is_repeatable_and_tracks_walk_length() -> None:
    neighbors = np.array(
        [
            [1, 2],
            [0, 2],
            [0, 1],
        ],
        dtype=np.int32,
    )
    graph = selection_eval.Graph(
        track_ids=np.array([1, 2, 3], dtype=np.int64),
        neighbors=neighbors,
        weights=np.full((3, 2), 0.5, dtype=np.float64),
    )
    transition = selection_eval.build_edge_transition(graph, weighted=False)
    first = selection_eval.exact_terminal_distribution(
        graph, seed=0, alpha=0.25, transition=transition, weighted=False
    )
    second = selection_eval.exact_terminal_distribution(
        graph, seed=0, alpha=0.25, transition=transition, weighted=False
    )

    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert first[0][0] == 0.0
    assert np.all(first[1][1:] > first[0][1:])


def test_monte_carlo_walk_is_reproducible_with_explicit_rng_seed() -> None:
    graph = selection_eval.Graph(
        track_ids=np.array([1, 2, 3], dtype=np.int64),
        neighbors=np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int32),
        weights=np.full((3, 2), 0.5, dtype=np.float64),
    )
    first = selection_eval.monte_carlo_terminal_counts(
        graph, seed=0, alpha=0.25, walks=500, rng=np.random.default_rng(123)
    )
    second = selection_eval.monte_carlo_terminal_counts(
        graph, seed=0, alpha=0.25, walks=500, rng=np.random.default_rng(123)
    )
    assert np.array_equal(first, second)
    assert first.sum() > 0
