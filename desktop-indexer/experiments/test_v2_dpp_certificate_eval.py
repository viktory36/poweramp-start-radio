from dataclasses import replace

import numpy as np

import v2_dpp_certificate_eval as certificate
import v2_selection_knob_matrix as matrix


def config(queue_size: int, exponent: float = 1.0, spacing: int = 0):
    return matrix.SelectorConfig(
        mode="dpp",
        queue_size=queue_size,
        dpp_exponent=exponent,
        artist_limits=spacing > 0,
        max_per_artist=8,
        artist_spacing=spacing,
    )


def identity(count: int) -> np.ndarray:
    return np.eye(count, dtype=np.float32)


def assert_matches_full(
    result: certificate.CertificateRun,
    embeddings: np.ndarray,
    relevance: np.ndarray,
    artists: np.ndarray,
    selector_config: matrix.SelectorConfig,
    valid_mask: np.ndarray | None = None,
) -> None:
    full = certificate.greedy_dpp(
        embeddings,
        relevance,
        artists,
        selector_config,
        valid_mask,
    )
    assert result.greedy.selected_indices == full.selected_indices


def test_strict_tie_expands_to_preserve_stable_candidate_order() -> None:
    embeddings = identity(2)
    relevance = np.array([1.0, 1.0], dtype=np.float32)
    artists = np.array([-1, -1], dtype=np.int32)
    selector_config = config(queue_size=1)

    result = certificate.select_certified(
        embeddings, relevance, artists, selector_config, initial_candidate_count=1
    )

    assert result.attempted_candidate_counts == (1, 2)
    assert result.greedy.selected_indices == (0,)
    assert_matches_full(result, embeddings, relevance, artists, selector_config)


def test_artist_spacing_can_reenable_an_earlier_artist() -> None:
    embeddings = identity(4)
    relevance = np.array([0.9, 0.85, 0.8, 0.1], dtype=np.float32)
    artists = np.array([0, 0, 1, 2], dtype=np.int32)
    selector_config = config(queue_size=3, spacing=1)

    result = certificate.select_certified(
        embeddings, relevance, artists, selector_config, initial_candidate_count=2
    )

    assert result.greedy.selected_indices == (0, 2, 1)
    assert result.attempted_candidate_counts == (2, 4)
    assert_matches_full(result, embeddings, relevance, artists, selector_config)


def test_missing_embedding_only_forces_conservative_expansion() -> None:
    embeddings = identity(3)
    relevance = np.array([1.0, 0.9, 0.8], dtype=np.float32)
    artists = np.array([-1, -1, -1], dtype=np.int32)
    valid = np.array([True, False, True])
    selector_config = config(queue_size=2)

    result = certificate.select_certified(
        embeddings,
        relevance,
        artists,
        selector_config,
        initial_candidate_count=1,
        valid_mask=valid,
    )

    assert result.attempted_candidate_counts == (1, 2, 3)
    assert result.greedy.selected_indices == (0, 2)
    assert_matches_full(result, embeddings, relevance, artists, selector_config, valid)


def test_unseen_below_stopping_threshold_certifies_short_queue() -> None:
    embeddings = identity(2)
    relevance = np.array([1.0, 1e-6], dtype=np.float32)
    artists = np.array([-1, -1], dtype=np.int32)
    selector_config = config(queue_size=2)

    result = certificate.select_certified(
        embeddings, relevance, artists, selector_config, initial_candidate_count=1
    )

    assert result.attempted_candidate_counts == (1,)
    assert result.final_unseen_gain_upper_bound is not None
    assert result.final_unseen_gain_upper_bound <= certificate.MIN_MARGINAL_GAIN
    assert result.greedy.selected_indices == (0,)
    assert_matches_full(result, embeddings, relevance, artists, selector_config)


def test_suffix_max_handles_non_monotonic_candidate_relevance() -> None:
    embeddings = identity(3)
    relevance = np.array([0.8, 0.1, 0.9], dtype=np.float32)
    artists = np.array([-1, -1, -1], dtype=np.int32)
    selector_config = config(queue_size=1)

    result = certificate.select_certified(
        embeddings, relevance, artists, selector_config, initial_candidate_count=1
    )

    assert result.attempted_candidate_counts == (1, 2, 3)
    assert result.greedy.selected_indices == (2,)
    assert_matches_full(result, embeddings, relevance, artists, selector_config)


def test_exponent_zero_gives_zero_relevance_unit_quality() -> None:
    scores = certificate.quality_scores(
        np.array([0.0, -1.0, 0.5], dtype=np.float32),
        0.0,
    )
    np.testing.assert_array_equal(scores, np.ones(3, dtype=np.float32))
