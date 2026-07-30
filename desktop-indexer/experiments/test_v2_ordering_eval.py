from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import v2_ordering_eval as ordering


def context(
    angles: list[float],
    artists: list[int],
    baseline: list[int] | None = None,
    track_ids: list[int] | None = None,
) -> ordering.OrderingContext:
    embeddings = np.asarray(
        [[np.cos(angle), np.sin(angle)] for angle in angles], dtype=np.float32
    )
    seed = np.asarray([1.0, 0.0], dtype=np.float32)
    pair_cosines = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    members = tuple(range(len(angles))) if baseline is None else tuple(baseline)
    local_embeddings = embeddings[np.asarray(members)]
    local_pair = np.clip(local_embeddings @ local_embeddings.T, -1.0, 1.0)
    return ordering.OrderingContext(
        membership=members,
        track_ids=np.asarray(track_ids or list(range(100, 100 + len(angles))), dtype=np.int64)[
            np.asarray(members)
        ],
        embeddings=local_embeddings,
        seed_embedding=seed,
        seed_cosines=np.clip(local_embeddings @ seed, -1.0, 1.0),
        pair_cosines=local_pair,
        pair_angles=np.arccos(local_pair.astype(np.float64)),
        artist_codes=np.asarray(artists, dtype=np.int32)[np.asarray(members)],
        spacing=3,
        max_per_artist=8,
    )


def test_all_variants_preserve_membership_and_spacing() -> None:
    ctx = context(
        [0.05, 0.2, 0.9, 1.6, 2.2, -0.4, -1.1, -2.0, 2.8, -2.8],
        [0, 1, 2, 3, 0, 4, 5, 6, 0, 7],
        baseline=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    algorithms = (
        ordering.current_order,
        ordering.constrained_ham1,
        ordering.constrained_ham2,
        ordering.seed_fixed_two_opt,
        ordering.seed_frontier,
    )
    for algorithm in algorithms:
        first = algorithm(ctx)
        second = algorithm(ctx)
        assert first == second
        assert sorted(first.order) == list(range(ctx.count))
        assert ordering.spacing_violations(ctx, first.order) == 0


def test_seed_fixed_variants_open_with_closest_member() -> None:
    ctx = context(
        [1.1, 0.6, 0.1, 1.8, -1.0, -2.0],
        [0, 1, 2, 3, 4, 5],
        track_ids=[30, 20, 10, 40, 50, 60],
    )
    closest = ordering.seed_ranked(ctx)[0]
    assert ordering.constrained_ham1(ctx).order[0] == closest
    assert ordering.seed_fixed_two_opt(ctx).order[0] == closest
    assert ordering.seed_frontier(ctx).order[0] == closest


def test_two_opt_never_increases_ham1_angular_cost() -> None:
    ctx = context(
        [0.1, 2.5, 0.4, -2.2, 1.1, -0.7, 2.9, -1.4],
        list(range(8)),
    )
    ham1 = ordering.constrained_ham1(ctx)
    optimized = ordering.seed_fixed_two_opt(ctx)
    assert ordering.path_angular_cost(ctx, optimized.order) <= (
        ordering.path_angular_cost(ctx, ham1.order) + ordering.EPSILON
    )


def test_track_id_is_canonical_tie_break() -> None:
    ctx = context(
        [0.0, 0.5, -0.5, 1.2],
        [0, 1, 2, 3],
        track_ids=[100, 300, 200, 400],
    )
    order = ordering.constrained_ham1(ctx).order
    assert order[:3] == (0, 2, 1)
