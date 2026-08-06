from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("v2_queue_eval.py")
SPEC = importlib.util.spec_from_file_location("v2_queue_eval", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
queue_eval = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue_eval
SPEC.loader.exec_module(queue_eval)


def make_library(artists: list[str | None]) -> queue_eval.Library:
    count = len(artists)
    # Unit vectors whose first coordinate gives a strict relevance order.
    angles = np.linspace(0.05, 1.0, count, dtype=np.float32)
    embeddings = np.zeros((count, count + 1), dtype=np.float32)
    embeddings[:, 0] = np.cos(angles)
    embeddings[np.arange(count), np.arange(count) + 1] = np.sin(angles)
    return queue_eval.Library(
        track_ids=np.arange(1, count + 1, dtype=np.int64),
        embeddings=embeddings,
        artists=tuple(artists),
        albums=tuple("album" for _ in artists),
        titles=tuple(f"track-{index}" for index in range(count)),
        durations_ms=np.full(count, 180_000, dtype=np.int64),
        clusters=np.zeros(count, dtype=np.int32),
        sources=tuple("desktop" for _ in artists),
        metadata_keys=tuple(f"metadata-{index}" for index in range(count)),
        filename_keys=tuple(f"filename-{index}" for index in range(count)),
        file_paths=tuple(f"file-{index}" for index in range(count)),
    )


def assert_artist_contract(
    library: queue_eval.Library,
    selected: list[int],
    max_per_artist: int,
    min_spacing: int,
) -> None:
    accepted: list[int] = []
    for index in selected:
        assert queue_eval.can_add_artist(
            index,
            accepted,
            library,
            max_per_artist,
            min_spacing,
        )
        accepted.append(index)


def test_post_filter_drops_without_backfill() -> None:
    library = make_library(["seed", "same", "same", "other", "third"])
    selected = [1, 2, 3]
    assert queue_eval.production_post_filter(selected, library, 8, 1) == [1, 3]


def test_constraint_aware_mmr_fills_and_is_deterministic() -> None:
    library = make_library(["seed", "same", "same", "other", "third", "fourth"])
    candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    relevance = library.embeddings[candidates] @ library.embeddings[0]

    first = queue_eval.select_mmr(
        library,
        candidates,
        relevance,
        count=4,
        lambda_=1.0,
        constraint_aware=True,
        max_per_artist=8,
        min_spacing=1,
    )
    second = queue_eval.select_mmr(
        library,
        candidates,
        relevance,
        count=4,
        lambda_=1.0,
        constraint_aware=True,
        max_per_artist=8,
        min_spacing=1,
    )

    assert first == second
    assert len(first[0]) == 4
    assert_artist_contract(library, first[0], max_per_artist=8, min_spacing=1)


def test_unknown_artist_consumes_a_spacing_position_without_being_blocked() -> None:
    library = make_library(["seed", "same", None, "same", "other"])
    candidates = np.array([1, 2, 3, 4], dtype=np.int64)
    relevance = library.embeddings[candidates] @ library.embeddings[0]
    selected, _ = queue_eval.select_mmr(
        library,
        candidates,
        relevance,
        count=4,
        lambda_=1.0,
        constraint_aware=True,
        max_per_artist=8,
        min_spacing=1,
    )
    assert selected[:3] == [1, 2, 3]


def test_constraint_aware_dpp_fills_artist_valid_queue() -> None:
    library = make_library(["seed", "same", "same", "other", "third", "fourth"])
    candidates = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    relevance = library.embeddings[candidates] @ library.embeddings[0]
    selected, _ = queue_eval.select_dpp(
        library,
        candidates,
        relevance,
        count=4,
        constraint_aware=True,
        max_per_artist=8,
        min_spacing=1,
    )
    assert len(selected) == 4
    assert_artist_contract(library, selected, max_per_artist=8, min_spacing=1)
