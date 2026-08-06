import numpy as np

import v2_dpp_adaptive_bounded_eval as adaptive_eval
import v2_dpp_memory_bounded_eval as memory_eval


def test_blocked_gather_is_exact_against_materialized() -> None:
    generator = np.random.default_rng(151)
    embeddings = generator.normal(size=(37, 19)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    candidates = np.array(
        [31, 4, 19, 2, 35, 7, 14, 28, 1, 22, 10, 16, 33], dtype=np.int64
    )
    relevance = (embeddings[candidates] @ embeddings[0]).astype(np.float32)
    artists = np.array([0, 1, 2, 1, 3, 4, 5, 0, 6, 7, 8, 9, 10], dtype=np.int32)

    materialized = memory_eval.greedy_materialized(
        embeddings, candidates, relevance, artists, exponent=1.0
    )
    bounded = adaptive_eval.greedy_bounded_gather(
        embeddings, candidates, relevance, artists, exponent=1.0, block_rows=4
    )

    assert bounded.selected_indices == materialized.selected_indices
    assert np.array_equal(
        np.asarray(bounded.selected_marginal_gains, dtype=np.float32),
        np.asarray(materialized.selected_marginal_gains, dtype=np.float32),
    )


def test_adaptive_counts_all_discarded_prefix_work_and_matches_full() -> None:
    embeddings = np.eye(4, dtype=np.float32)
    candidates = np.arange(4, dtype=np.int64)
    relevance = np.array([1.0, 0.9, 0.8, 0.7], dtype=np.float32)
    artists = np.full(4, -1, dtype=np.int32)

    adaptive = adaptive_eval.select_certified_bounded(
        embeddings,
        candidates,
        relevance,
        artists,
        exponent=1.0,
        initial_candidate_count=1,
        growth_factor=2.0,
        block_rows=2,
    )
    full = adaptive_eval.greedy_bounded_gather(
        embeddings, candidates, relevance, artists, exponent=1.0, block_rows=2
    )

    assert adaptive.attempted_candidate_counts == (1, 2, 4)
    assert adaptive.selected_steps_per_attempt == (1, 2, 4)
    assert adaptive.candidate_step_rows == 1 * 1 + 2 * 2 + 4 * 4
    assert adaptive.greedy == full
