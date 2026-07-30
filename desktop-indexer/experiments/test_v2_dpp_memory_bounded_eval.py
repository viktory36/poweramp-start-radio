import numpy as np

import v2_dpp_memory_bounded_eval as memory_eval


def normalized_rows(seed: int, count: int, dim: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(count, dim)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values.astype(np.float32, copy=False)


def test_streaming_similarity_columns_match_materialized_greedy() -> None:
    embeddings = normalized_rows(seed=14, count=31, dim=16)
    candidates = np.array([19, 2, 25, 8, 13, 4, 29, 6, 1, 17, 23], dtype=np.int64)
    relevance = (embeddings[candidates] @ embeddings[0]).astype(np.float32)
    artists = np.array([0, 1, 1, 2, 3, 0, 4, 5, 6, 7, 8], dtype=np.int32)

    materialized = memory_eval.greedy_materialized(
        embeddings, candidates, relevance, artists, exponent=0.5
    )
    streaming = memory_eval.greedy_streaming(
        embeddings, candidates, relevance, artists, exponent=0.5
    )

    assert materialized.selected_indices == streaming.selected_indices
    assert np.array_equal(
        np.asarray(materialized.selected_marginal_gains, dtype=np.float32),
        np.asarray(streaming.selected_marginal_gains, dtype=np.float32),
    )


def test_exponent_zero_preserves_stable_candidate_order_for_orthogonal_rows() -> None:
    embeddings = np.eye(6, dtype=np.float32)
    candidates = np.array([5, 3, 1, 4, 2], dtype=np.int64)
    relevance = np.zeros(candidates.size, dtype=np.float32)
    artists = np.full(candidates.size, -1, dtype=np.int32)

    result = memory_eval.greedy_streaming(
        embeddings, candidates, relevance, artists, exponent=0.0
    )

    assert result.selected_indices == (0, 1, 2, 3, 4)
    assert result.selected_marginal_gains == (1.0, 1.0, 1.0, 1.0, 1.0)
