#!/usr/bin/env python3
"""Falsify or advance identity-safe local audio facets on the active library.

Clustering, ranking, and every geometry metric use CLaMP3 embeddings only.  The only
rows quotiented together are pairs independently proven to have byte-identical
decoded PCM.  Artist strings are never clustering features: they are used solely
to audit catalog domination and to choose readable display representatives after
the embedding-only partition has already been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_capability_eval as capability
import v2_dedupe_pool_eval as dedupe
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATABASE = matrix.DEFAULT_DATABASE
DEFAULT_ACTIVE_CATALOG = matrix.DEFAULT_ACTIVE_CATALOG
DEFAULT_AUDIO_IDENTITY = dedupe.DEFAULT_AUDIO_IDENTITY
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "identity-safe-local-facets-2026-07-15"
)

EXPERIMENT_VERSION = "identity-safe-local-facets-active-domain-v1"
PRIOR_FACET_SEED_IDS = (13384, 209, 25130, 21972, 2682, 5831, 31830, 74859)
# This was an independently frozen real-device cohort before the facet experiment.
EXTRA_SEED_IDS = (80437, 42335, 38327, 33821, 17399, 5987, 35389, 79210)
SEED_IDS = PRIOR_FACET_SEED_IDS + EXTRA_SEED_IDS
NEIGHBORHOOD_WIDTHS = (100, 200, 500)
BRANCH_COUNTS = (3, 4, 5)
ANCHOR_WIDTH = 200
ANCHOR_BRANCHES = 4
QUEUE_SIZE = 30
PREVIEW_COUNT = 6
INFLUENCE_COUNT = 30
PACKET_SEED_IDS = (13384, 209, 2682, 74859, 80437, 33821, 5987, 79210)

# Frozen before execution. Passing these gates advances the interaction to listening;
# it does not establish that the directions sound useful and is not a shipping gate.
LISTENING_GATES = {
    "all_repeat_deterministic": True,
    "exact_copy_leakage": 0,
    "anchor_median_silhouette_min": 0.0,
    "anchor_median_balance_entropy_min": 0.75,
    "anchor_median_min_branch_fraction_min": 0.04,
    "k4_adjacent_width_median_ari_min": 0.45,
    "k4_adjacent_width_median_matched_medoid_cosine_min": 0.95,
    "anchor_median_primary_artist_distinct_fraction_min": 1.0,
    "anchor_median_preview_artist_distinct_fraction_min": 0.75,
    "anchor_median_primary_display_cosine_min": 0.90,
    "anchor_facility_win_rate_vs_raw_min": 0.75,
    "anchor_facility_win_rate_vs_mmr_min": 0.50,
    "anchor_facility_win_rate_vs_dpp_min": 0.50,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=queue_eval.json_numpy_scalar,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            default=queue_eval.json_numpy_scalar,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def mean(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return float(np.mean(materialized)) if materialized else None


def median(values: Iterable[float]) -> float | None:
    materialized = [float(value) for value in values]
    return float(np.median(materialized)) if materialized else None


def row_label(library: queue_eval.Library, index: int) -> dict[str, object]:
    return {
        "track_id": int(library.track_ids[index]),
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "duration_ms": int(library.durations_ms[index]),
        "file_path": library.file_paths[index],
    }


def normalized_artist_token(library: queue_eval.Library, index: int) -> str:
    artist = matrix.normalized_artist(library.artists[index])
    if artist is not None:
        return artist
    return f"__unknown_track_{int(library.track_ids[index])}"


def quotient_proven_pcm_identities(
    active: queue_eval.Library,
    identity_path: Path,
) -> tuple[queue_eval.Library, list[dict[str, object]]]:
    evidence = json.loads(identity_path.read_text(encoding="utf-8"))
    active_by_id = {int(track_id): index for index, track_id in enumerate(active.track_ids)}
    groups: list[list[int]] = []
    for record in evidence["records"]:
        if not bool(record.get("evidence", {}).get("decoded_pcm_identical")):
            continue
        pair = [int(record["left"]["track_id"]), int(record["right"]["track_id"])]
        if all(track_id in active_by_id for track_id in pair):
            groups.append(pair)

    # Merge defensively in case a later independently proven pair overlaps a group.
    merged: list[set[int]] = []
    for pair in groups:
        matching = [group for group in merged if group.intersection(pair)]
        if not matching:
            merged.append(set(pair))
            continue
        union = set(pair)
        for group in matching:
            union.update(group)
            merged.remove(group)
        merged.append(union)

    dropped: set[int] = set()
    audit: list[dict[str, object]] = []
    mean_embedding_by_canonical: dict[int, np.ndarray] = {}
    for group in sorted(merged, key=lambda values: min(values)):
        members = sorted(group)
        canonical_id = members[0]
        indices = np.asarray([active_by_id[track_id] for track_id in members], dtype=np.int64)
        vector = np.mean(active.embeddings[indices].astype(np.float64), axis=0)
        vector /= np.linalg.norm(vector)
        mean_embedding_by_canonical[canonical_id] = vector.astype(np.float32)
        dropped.update(members[1:])
        audit.append(
            {
                "canonical_track_id": canonical_id,
                "member_track_ids": members,
                "representative_policy": "lowest track ID; identity vector is normalized member mean",
            }
        )

    keep = np.fromiter(
        (
            index
            for index, track_id in enumerate(active.track_ids)
            if int(track_id) not in dropped
        ),
        dtype=np.int64,
        count=active.count - len(dropped),
    )
    quotient = matrix.subset_library(active, keep)
    embeddings = quotient.embeddings.copy()
    quotient_by_id = {
        int(track_id): index for index, track_id in enumerate(quotient.track_ids)
    }
    for canonical_id, vector in mean_embedding_by_canonical.items():
        embeddings[quotient_by_id[canonical_id]] = vector
    quotient = replace(quotient, embeddings=embeddings)
    return quotient, audit


def reorder_branches(
    medoids: np.ndarray,
    assignments: np.ndarray,
    seed_cosines: np.ndarray,
    local_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    branch_order = sorted(
        range(medoids.size),
        key=lambda branch: (
            -float(seed_cosines[int(medoids[branch])]),
            int(local_ids[int(medoids[branch])]),
        ),
    )
    old_to_new = {old: new for new, old in enumerate(branch_order)}
    return (
        medoids[np.asarray(branch_order, dtype=np.int64)],
        np.fromiter(
            (old_to_new[int(value)] for value in assignments),
            dtype=np.int64,
            count=assignments.size,
        ),
    )


def artist_constrained_display(
    library: queue_eval.Library,
    local: np.ndarray,
    medoids: np.ndarray,
    assignments: np.ndarray,
) -> tuple[list[int], list[list[int]], dict[str, float | int]]:
    branch_count = int(medoids.size)
    candidate_by_branch_artist: list[dict[str, tuple[int, float]]] = []
    all_artists: set[str] = set()
    ordered_by_branch: list[list[int]] = []
    for branch in range(branch_count):
        member_local = np.flatnonzero(assignments == branch)
        member_global = local[member_local]
        medoid_global = int(local[int(medoids[branch])])
        similarities = library.embeddings[member_global] @ library.embeddings[medoid_global]
        order = np.lexsort((library.track_ids[member_global], -similarities))
        ordered = [int(value) for value in member_global[order]]
        ordered_by_branch.append(ordered)
        best: dict[str, tuple[int, float]] = {}
        for index, similarity in zip(member_global[order], similarities[order]):
            raw_index = int(index)
            token = normalized_artist_token(library, raw_index)
            best.setdefault(token, (raw_index, float(similarity)))
        candidate_by_branch_artist.append(best)
        all_artists.update(best)

    columns = sorted(all_artists)
    scores = np.full((branch_count, len(columns)), -1_000_000.0, dtype=np.float64)
    for branch, choices in enumerate(candidate_by_branch_artist):
        for column, artist in enumerate(columns):
            if artist in choices:
                scores[branch, column] = choices[artist][1]
    rows, chosen_columns = linear_sum_assignment(-scores)
    primary = [-1] * branch_count
    for branch, column in zip(rows, chosen_columns):
        choice = candidate_by_branch_artist[int(branch)].get(columns[int(column)])
        if choice is not None:
            primary[int(branch)] = choice[0]
    for branch in range(branch_count):
        if primary[branch] < 0:
            primary[branch] = ordered_by_branch[branch][0]

    previews: list[list[int]] = []
    for branch, ordered in enumerate(ordered_by_branch):
        chosen = [primary[branch]]
        used = {normalized_artist_token(library, primary[branch])}
        for index in ordered:
            if index in chosen:
                continue
            token = normalized_artist_token(library, index)
            if token in used:
                continue
            chosen.append(index)
            used.add(token)
            if len(chosen) == PREVIEW_COUNT:
                break
        if len(chosen) < PREVIEW_COUNT:
            for index in ordered:
                if index not in chosen:
                    chosen.append(index)
                if len(chosen) == PREVIEW_COUNT:
                    break
        previews.append(chosen)

    primary_tokens = [normalized_artist_token(library, index) for index in primary]
    preview_tokens = [
        normalized_artist_token(library, index) for branch in previews for index in branch
    ]
    primary_cosines = [
        float(library.embeddings[primary[branch]] @ library.embeddings[int(local[medoids[branch]])])
        for branch in range(branch_count)
    ]
    metrics: dict[str, float | int] = {
        "primary_distinct_artist_fraction": len(set(primary_tokens)) / len(primary_tokens),
        "preview_distinct_artist_fraction": len(set(preview_tokens)) / len(preview_tokens),
        "primary_replacement_count": sum(
            primary[branch] != int(local[medoids[branch]]) for branch in range(branch_count)
        ),
        "mean_primary_medoid_cosine": float(np.mean(primary_cosines)),
        "minimum_primary_medoid_cosine": float(np.min(primary_cosines)),
    }
    return primary, previews, metrics


def pairwise_jaccards(sets: Sequence[set[int]]) -> list[float]:
    values: list[float] = []
    for left in range(len(sets)):
        for right in range(left + 1, len(sets)):
            union = sets[left] | sets[right]
            values.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
    return values


def pairwise_upper(matrix_: np.ndarray) -> np.ndarray:
    return matrix_[np.triu_indices(matrix_.shape[0], 1)]


def cluster_record(
    library: queue_eval.Library,
    seed_index: int,
    context: matrix.SeedContext,
    width: int,
    branch_count: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    local = context.closest_order[:width].copy()
    embeddings = library.embeddings[local]
    local_ids = library.track_ids[local]
    medoids, assignments, iterations = capability.deterministic_kmedoids(
        embeddings, local_ids, branch_count
    )
    repeat_medoids, repeat_assignments, _ = capability.deterministic_kmedoids(
        embeddings, local_ids, branch_count
    )
    deterministic = bool(
        np.array_equal(medoids, repeat_medoids)
        and np.array_equal(assignments, repeat_assignments)
    )
    medoids, assignments = reorder_branches(
        medoids,
        assignments,
        context.seed_similarities[local],
        local_ids,
    )

    gram = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distance = np.maximum(0.0, 1.0 - gram)
    np.fill_diagonal(distance, 0.0)
    silhouette = float(silhouette_score(distance, assignments, metric="precomputed"))
    sizes = np.bincount(assignments, minlength=branch_count)
    fractions = sizes.astype(np.float64) / sizes.sum()
    entropy = float(-np.sum(fractions * np.log(fractions)) / math.log(branch_count))
    same = assignments[:, None] == assignments[None, :]
    upper = np.triu(np.ones_like(same, dtype=np.bool_), 1)
    within = gram[upper & same]
    between = gram[upper & ~same]

    centroid_vectors: list[np.ndarray] = []
    branch_records: list[dict[str, object]] = []
    unconstrained_preview_tokens: list[str] = []
    influence_sets: list[set[int]] = []
    for branch in range(branch_count):
        member_local = np.flatnonzero(assignments == branch)
        member_global = local[member_local]
        medoid_global = int(local[int(medoids[branch])])
        centroid = np.mean(library.embeddings[member_global].astype(np.float64), axis=0)
        centroid /= np.linalg.norm(centroid)
        centroid_vectors.append(centroid)
        medoid_similarities = embeddings @ library.embeddings[medoid_global]
        influence_order = np.lexsort((local_ids, -medoid_similarities))
        influence_sets.append(
            set(int(value) for value in local_ids[influence_order[:INFLUENCE_COUNT]])
        )
        member_similarities = (
            library.embeddings[member_global] @ library.embeddings[medoid_global]
        )
        member_order = np.lexsort(
            (library.track_ids[member_global], -member_similarities)
        )
        raw_previews = [int(value) for value in member_global[member_order[:PREVIEW_COUNT]]]
        unconstrained_preview_tokens.extend(
            normalized_artist_token(library, index) for index in raw_previews
        )
        artist_counts = Counter(
            normalized_artist_token(library, int(index)) for index in member_global
        )
        branch_records.append(
            {
                "branch": branch + 1,
                "size": int(member_global.size),
                "medoid": row_label(library, medoid_global),
                "mean_seed_cosine": float(
                    np.mean(context.seed_similarities[member_global])
                ),
                "minimum_seed_cosine": float(
                    np.min(context.seed_similarities[member_global])
                ),
                "membership_max_artist_share": max(artist_counts.values())
                / member_global.size,
                "membership_distinct_artists": len(artist_counts),
                "unconstrained_previews": [
                    row_label(library, index) for index in raw_previews
                ],
            }
        )

    centroid_gram = np.asarray(centroid_vectors) @ np.asarray(centroid_vectors).T
    medoid_gram = (
        library.embeddings[local[medoids]] @ library.embeddings[local[medoids]].T
    )
    influence_jaccards = pairwise_jaccards(influence_sets)
    primary, previews, display_metrics = artist_constrained_display(
        library, local, medoids, assignments
    )
    for branch in range(branch_count):
        branch_records[branch]["display_representative"] = row_label(
            library, primary[branch]
        )
        branch_records[branch]["display_previews"] = [
            row_label(library, index) for index in previews[branch]
        ]

    record: dict[str, object] = {
        "seed": row_label(library, seed_index),
        "width": width,
        "branch_count": branch_count,
        "iterations": iterations,
        "repeat_deterministic": deterministic,
        "local_track_ids": [int(value) for value in local_ids],
        "assignments": [int(value) + 1 for value in assignments],
        "medoid_track_ids": [int(value) for value in library.track_ids[local[medoids]]],
        "display_representative_track_ids": [
            int(library.track_ids[index]) for index in primary
        ],
        "display_preview_track_ids": [
            [int(library.track_ids[index]) for index in branch] for branch in previews
        ],
        "geometry": {
            "silhouette_cosine": silhouette,
            "mean_within_cosine": float(np.mean(within)),
            "mean_between_cosine": float(np.mean(between)),
            "within_minus_between_cosine": float(np.mean(within) - np.mean(between)),
            "mean_medoid_pair_cosine": float(np.mean(pairwise_upper(medoid_gram))),
            "maximum_medoid_pair_cosine": float(np.max(pairwise_upper(medoid_gram))),
            "mean_centroid_pair_cosine": float(np.mean(pairwise_upper(centroid_gram))),
            "maximum_centroid_pair_cosine": float(np.max(pairwise_upper(centroid_gram))),
            "mean_cross_branch_influence_jaccard": float(np.mean(influence_jaccards)),
            "maximum_cross_branch_influence_jaccard": float(np.max(influence_jaccards)),
        },
        "balance": {
            "sizes": [int(value) for value in sizes],
            "normalized_entropy": entropy,
            "minimum_fraction": float(np.min(fractions)),
            "maximum_fraction": float(np.max(fractions)),
        },
        "artist_display_audit": {
            **display_metrics,
            "unconstrained_preview_distinct_artist_fraction": len(
                set(unconstrained_preview_tokens)
            )
            / len(unconstrained_preview_tokens),
            "mean_membership_max_artist_share": float(
                np.mean(
                    [branch["membership_max_artist_share"] for branch in branch_records]
                )
            ),
            "maximum_membership_max_artist_share": float(
                np.max(
                    [branch["membership_max_artist_share"] for branch in branch_records]
                )
            ),
        },
        "branches": branch_records,
    }
    internal = {
        "local": local,
        "local_ids": local_ids,
        "assignments": assignments,
        "medoids": local[medoids],
        "primary": np.asarray(primary, dtype=np.int64),
        "previews": np.asarray(
            [index for branch in previews for index in branch], dtype=np.int64
        ),
    }
    return record, internal


def baseline_queues(
    library: queue_eval.Library,
    context: matrix.SeedContext,
) -> tuple[dict[str, dict[str, object]], dict[str, np.ndarray]]:
    result: dict[str, dict[str, object]] = {}
    internal: dict[str, np.ndarray] = {}

    raw_config = matrix.SelectorConfig(mode="closest", queue_size=QUEUE_SIZE)
    started = time.perf_counter()
    raw, ranks, scores, pool = matrix.select_closest(library, context, raw_config)
    result["raw_nearest"] = {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "candidate_count": pool,
        "candidate_ranks": ranks,
        "seed_cosines": scores,
        "tracks": [row_label(library, index) for index in raw],
    }
    internal["raw_nearest"] = np.asarray(raw, dtype=np.int64)

    mmr_config = matrix.SelectorConfig(mode="mmr", queue_size=QUEUE_SIZE)
    candidates, relevance, pool = matrix.candidates_for(library, context, mmr_config)
    started = time.perf_counter()
    mmr, ranks, scores = matrix.select_mmr(library, candidates, relevance, mmr_config)
    result["mmr"] = {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "candidate_count": pool,
        "candidate_ranks": ranks,
        "seed_cosines": scores,
        "tracks": [row_label(library, index) for index in mmr],
    }
    internal["mmr"] = np.asarray(mmr, dtype=np.int64)

    dpp_config = matrix.SelectorConfig(
        mode="dpp",
        queue_size=QUEUE_SIZE,
        dpp_uses_certified_full_domain=True,
    )
    candidates, relevance, pool = matrix.candidates_for(library, context, dpp_config)
    started = time.perf_counter()
    dpp, ranks, scores = matrix.select_dpp(library, candidates, relevance, dpp_config)
    result["dpp"] = {
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "candidate_count": pool,
        "candidate_ranks": ranks,
        "seed_cosines": scores,
        "tracks": [row_label(library, index) for index in dpp],
    }
    internal["dpp"] = np.asarray(dpp, dtype=np.int64)
    return result, internal


def representation_comparison(
    library: queue_eval.Library,
    internal: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
) -> dict[str, object]:
    local = internal["local"]
    local_embeddings = library.embeddings[local]
    assignments = internal["assignments"]
    branch_count = int(internal["medoids"].size)
    preview_ids = set(int(library.track_ids[index]) for index in internal["previews"])

    def metrics(representatives: np.ndarray, queue: np.ndarray) -> dict[str, object]:
        reps = representatives[:branch_count]
        similarities = local_embeddings @ library.embeddings[reps].T
        gram = library.embeddings[reps] @ library.embeddings[reps].T
        local_lookup = {
            int(global_index): int(assignments[local_index])
            for local_index, global_index in enumerate(local)
        }
        in_local_branches = [local_lookup[int(index)] for index in queue if int(index) in local_lookup]
        counts = Counter(in_local_branches)
        fractions = np.asarray(list(counts.values()), dtype=np.float64)
        entropy = None
        if fractions.size > 1:
            fractions /= fractions.sum()
            entropy = float(-np.sum(fractions * np.log(fractions)) / math.log(branch_count))
        queue_ids = set(int(library.track_ids[index]) for index in queue)
        union = preview_ids | queue_ids
        return {
            "facility_coverage_mean_max_cosine": float(np.mean(np.max(similarities, axis=1))),
            "mean_representative_pair_cosine": float(np.mean(pairwise_upper(gram))),
            "maximum_representative_pair_cosine": float(np.max(pairwise_upper(gram))),
            "queue_items_inside_local_neighborhood": len(in_local_branches),
            "local_branches_represented_by_queue": len(counts),
            "local_branch_allocation": [int(counts.get(branch, 0)) for branch in range(branch_count)],
            "local_branch_allocation_entropy": entropy,
            "display_preview_queue_jaccard": len(preview_ids & queue_ids) / len(union)
            if union
            else 1.0,
        }

    medoids = internal["medoids"]
    primary = internal["primary"]
    comparisons: dict[str, object] = {
        "facet_medoids": metrics(medoids, medoids),
        "display_representatives": metrics(primary, primary),
    }
    for name, queue in baselines.items():
        comparisons[name] = metrics(queue, queue)
    facet_coverage = comparisons["facet_medoids"]["facility_coverage_mean_max_cosine"]
    comparisons["facet_facility_delta"] = {
        name: facet_coverage - comparisons[name]["facility_coverage_mean_max_cosine"]
        for name in baselines
    }
    return comparisons


def matched_cluster_metrics(
    ids_left: np.ndarray,
    labels_left: np.ndarray,
    ids_right: np.ndarray,
    labels_right: np.ndarray,
    count_left: int,
    count_right: int,
) -> dict[str, float]:
    right_by_id = {int(track_id): index for index, track_id in enumerate(ids_right)}
    common_left = [index for index, track_id in enumerate(ids_left) if int(track_id) in right_by_id]
    common_right = [right_by_id[int(ids_left[index])] for index in common_left]
    left = labels_left[np.asarray(common_left, dtype=np.int64)]
    right = labels_right[np.asarray(common_right, dtype=np.int64)]
    intersections = np.zeros((count_left, count_right), dtype=np.float64)
    unions = np.zeros_like(intersections)
    for a in range(count_left):
        left_members = left == a
        for b in range(count_right):
            right_members = right == b
            intersections[a, b] = np.sum(left_members & right_members)
            unions[a, b] = np.sum(left_members | right_members)
    jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections),
        where=unions > 0,
    )
    rows, columns = linear_sum_assignment(-jaccard)
    return {
        "adjusted_rand": float(adjusted_rand_score(left, right)),
        "normalized_mutual_information": float(normalized_mutual_info_score(left, right)),
        "mean_matched_membership_jaccard": float(np.mean(jaccard[rows, columns])),
    }


def matched_medoid_metrics(
    library: queue_eval.Library,
    medoids_left: np.ndarray,
    medoids_right: np.ndarray,
) -> dict[str, float]:
    cosine = library.embeddings[medoids_left] @ library.embeddings[medoids_right].T
    rows, columns = linear_sum_assignment(-cosine)
    left_ids = library.track_ids[medoids_left[rows]]
    right_ids = library.track_ids[medoids_right[columns]]
    return {
        "mean_matched_medoid_cosine": float(np.mean(cosine[rows, columns])),
        "minimum_matched_medoid_cosine": float(np.min(cosine[rows, columns])),
        "exact_matched_medoid_fraction": float(np.mean(left_ids == right_ids)),
    }


def stability_results(
    library: queue_eval.Library,
    internals: dict[tuple[int, int, int], dict[str, np.ndarray]],
) -> dict[str, list[dict[str, object]]]:
    widths: list[dict[str, object]] = []
    for seed_id in SEED_IDS:
        for branches in BRANCH_COUNTS:
            for left_width, right_width in ((100, 200), (200, 500), (100, 500)):
                left = internals[(seed_id, left_width, branches)]
                right = internals[(seed_id, right_width, branches)]
                widths.append(
                    {
                        "seed_track_id": seed_id,
                        "branch_count": branches,
                        "left_width": left_width,
                        "right_width": right_width,
                        **matched_cluster_metrics(
                            left["local_ids"],
                            left["assignments"],
                            right["local_ids"],
                            right["assignments"],
                            branches,
                            branches,
                        ),
                        **matched_medoid_metrics(
                            library, left["medoids"], right["medoids"]
                        ),
                    }
                )

    branch_counts: list[dict[str, object]] = []
    for seed_id in SEED_IDS:
        for width in NEIGHBORHOOD_WIDTHS:
            for coarse_count, fine_count in ((3, 4), (4, 5), (3, 5)):
                coarse = internals[(seed_id, width, coarse_count)]
                fine = internals[(seed_id, width, fine_count)]
                common = matched_cluster_metrics(
                    coarse["local_ids"],
                    coarse["assignments"],
                    fine["local_ids"],
                    fine["assignments"],
                    coarse_count,
                    fine_count,
                )
                intersections = np.zeros((coarse_count, fine_count), dtype=np.int64)
                for coarse_branch in range(coarse_count):
                    for fine_branch in range(fine_count):
                        intersections[coarse_branch, fine_branch] = np.sum(
                            (coarse["assignments"] == coarse_branch)
                            & (fine["assignments"] == fine_branch)
                        )
                refinement = float(np.sum(np.max(intersections, axis=0)) / width)
                branch_counts.append(
                    {
                        "seed_track_id": seed_id,
                        "width": width,
                        "coarse_branch_count": coarse_count,
                        "fine_branch_count": fine_count,
                        **common,
                        "fine_to_coarse_refinement_purity": refinement,
                        **matched_medoid_metrics(
                            library, coarse["medoids"], fine["medoids"]
                        ),
                    }
                )
    return {"width": widths, "branch_count": branch_counts}


def summarize(
    records: list[dict[str, object]],
    stability: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    by_config: list[dict[str, object]] = []
    for width in NEIGHBORHOOD_WIDTHS:
        for branches in BRANCH_COUNTS:
            selected = [
                record
                for record in records
                if record["width"] == width and record["branch_count"] == branches
            ]
            by_config.append(
                {
                    "width": width,
                    "branch_count": branches,
                    "seeds": len(selected),
                    "deterministic_fraction": mean(
                        float(record["repeat_deterministic"]) for record in selected
                    ),
                    "median_silhouette": median(
                        record["geometry"]["silhouette_cosine"] for record in selected
                    ),
                    "median_balance_entropy": median(
                        record["balance"]["normalized_entropy"] for record in selected
                    ),
                    "median_minimum_branch_fraction": median(
                        record["balance"]["minimum_fraction"] for record in selected
                    ),
                    "median_maximum_branch_fraction": median(
                        record["balance"]["maximum_fraction"] for record in selected
                    ),
                    "median_influence_jaccard": median(
                        record["geometry"]["mean_cross_branch_influence_jaccard"]
                        for record in selected
                    ),
                    "median_membership_max_artist_share": median(
                        record["artist_display_audit"]["maximum_membership_max_artist_share"]
                        for record in selected
                    ),
                    "median_unconstrained_preview_artist_distinct_fraction": median(
                        record["artist_display_audit"][
                            "unconstrained_preview_distinct_artist_fraction"
                        ]
                        for record in selected
                    ),
                    "median_constrained_preview_artist_distinct_fraction": median(
                        record["artist_display_audit"]["preview_distinct_artist_fraction"]
                        for record in selected
                    ),
                    "median_primary_artist_distinct_fraction": median(
                        record["artist_display_audit"][
                            "primary_distinct_artist_fraction"
                        ]
                        for record in selected
                    ),
                    "median_primary_display_cosine": median(
                        record["artist_display_audit"]["mean_primary_medoid_cosine"]
                        for record in selected
                    ),
                    "facility_win_rate_vs_raw": mean(
                        float(
                            record["representation_comparison"]["facet_facility_delta"][
                                "raw_nearest"
                            ]
                            > 0.0
                        )
                        for record in selected
                    ),
                    "facility_win_rate_vs_mmr": mean(
                        float(
                            record["representation_comparison"]["facet_facility_delta"][
                                "mmr"
                            ]
                            > 0.0
                        )
                        for record in selected
                    ),
                    "facility_win_rate_vs_dpp": mean(
                        float(
                            record["representation_comparison"]["facet_facility_delta"][
                                "dpp"
                            ]
                            > 0.0
                        )
                        for record in selected
                    ),
                }
            )

    width_summary: list[dict[str, object]] = []
    for branches in BRANCH_COUNTS:
        for left, right in ((100, 200), (200, 500), (100, 500)):
            selected = [
                record
                for record in stability["width"]
                if record["branch_count"] == branches
                and record["left_width"] == left
                and record["right_width"] == right
            ]
            width_summary.append(
                {
                    "branch_count": branches,
                    "left_width": left,
                    "right_width": right,
                    "median_adjusted_rand": median(
                        record["adjusted_rand"] for record in selected
                    ),
                    "median_nmi": median(
                        record["normalized_mutual_information"] for record in selected
                    ),
                    "median_membership_jaccard": median(
                        record["mean_matched_membership_jaccard"] for record in selected
                    ),
                    "median_matched_medoid_cosine": median(
                        record["mean_matched_medoid_cosine"] for record in selected
                    ),
                    "median_exact_medoid_fraction": median(
                        record["exact_matched_medoid_fraction"] for record in selected
                    ),
                }
            )
    return {"by_config": by_config, "width_stability": width_summary}


def gate_decision(
    records: list[dict[str, object]],
    summary: dict[str, object],
    exact_copy_leakage: int,
) -> dict[str, object]:
    anchor = next(
        row
        for row in summary["by_config"]
        if row["width"] == ANCHOR_WIDTH and row["branch_count"] == ANCHOR_BRANCHES
    )
    adjacent = [
        row
        for row in summary["width_stability"]
        if row["branch_count"] == 4
        and (row["left_width"], row["right_width"]) in ((100, 200), (200, 500))
    ]
    observed = {
        "all_repeat_deterministic": all(record["repeat_deterministic"] for record in records),
        "exact_copy_leakage": exact_copy_leakage,
        "anchor_median_silhouette": anchor["median_silhouette"],
        "anchor_median_balance_entropy": anchor["median_balance_entropy"],
        "anchor_median_min_branch_fraction": anchor["median_minimum_branch_fraction"],
        "k4_adjacent_width_median_ari": median(
            row["median_adjusted_rand"] for row in adjacent
        ),
        "k4_adjacent_width_median_matched_medoid_cosine": median(
            row["median_matched_medoid_cosine"] for row in adjacent
        ),
        "anchor_median_primary_artist_distinct_fraction": anchor[
            "median_primary_artist_distinct_fraction"
        ],
        "anchor_median_preview_artist_distinct_fraction": anchor[
            "median_constrained_preview_artist_distinct_fraction"
        ],
        "anchor_median_primary_display_cosine": anchor[
            "median_primary_display_cosine"
        ],
        "anchor_facility_win_rate_vs_raw": anchor["facility_win_rate_vs_raw"],
        "anchor_facility_win_rate_vs_mmr": anchor["facility_win_rate_vs_mmr"],
        "anchor_facility_win_rate_vs_dpp": anchor["facility_win_rate_vs_dpp"],
    }
    checks = {
        "all_repeat_deterministic": observed["all_repeat_deterministic"] is True,
        "exact_copy_leakage": observed["exact_copy_leakage"] == 0,
        "anchor_median_silhouette": observed["anchor_median_silhouette"]
        >= LISTENING_GATES["anchor_median_silhouette_min"],
        "anchor_median_balance_entropy": observed["anchor_median_balance_entropy"]
        >= LISTENING_GATES["anchor_median_balance_entropy_min"],
        "anchor_median_min_branch_fraction": observed[
            "anchor_median_min_branch_fraction"
        ]
        >= LISTENING_GATES["anchor_median_min_branch_fraction_min"],
        "k4_adjacent_width_median_ari": observed["k4_adjacent_width_median_ari"]
        >= LISTENING_GATES["k4_adjacent_width_median_ari_min"],
        "k4_adjacent_width_median_matched_medoid_cosine": observed[
            "k4_adjacent_width_median_matched_medoid_cosine"
        ]
        >= LISTENING_GATES["k4_adjacent_width_median_matched_medoid_cosine_min"],
        "anchor_median_primary_artist_distinct_fraction": observed[
            "anchor_median_primary_artist_distinct_fraction"
        ]
        >= LISTENING_GATES["anchor_median_primary_artist_distinct_fraction_min"],
        "anchor_median_preview_artist_distinct_fraction": observed[
            "anchor_median_preview_artist_distinct_fraction"
        ]
        >= LISTENING_GATES["anchor_median_preview_artist_distinct_fraction_min"],
        "anchor_median_primary_display_cosine": observed[
            "anchor_median_primary_display_cosine"
        ]
        >= LISTENING_GATES["anchor_median_primary_display_cosine_min"],
        "anchor_facility_win_rate_vs_raw": observed["anchor_facility_win_rate_vs_raw"]
        >= LISTENING_GATES["anchor_facility_win_rate_vs_raw_min"],
        "anchor_facility_win_rate_vs_mmr": observed["anchor_facility_win_rate_vs_mmr"]
        >= LISTENING_GATES["anchor_facility_win_rate_vs_mmr_min"],
        "anchor_facility_win_rate_vs_dpp": observed["anchor_facility_win_rate_vs_dpp"]
        >= LISTENING_GATES["anchor_facility_win_rate_vs_dpp_min"],
    }
    return {
        "decision": "GO_TO_LISTENING_PROTOTYPE" if all(checks.values()) else "NO_GO",
        "scope": "structural gate only; human listening remains mandatory",
        "frozen_thresholds": LISTENING_GATES,
        "observed": observed,
        "checks": checks,
    }


def write_blind_packet(
    output: Path,
    library: queue_eval.Library,
    records_by_key: dict[tuple[int, int, int], dict[str, object]],
    baseline_records: dict[int, dict[str, dict[str, object]]],
) -> None:
    lines = [
        "# Blind Local-Facet Listening Packet",
        "",
        "For each seed, listen to the four directions and decide whether each is coherent,",
        "meaningfully different, and a direction you might deliberately choose. The three flat",
        "queues are algorithm-blinded. Do not open `blind_key.json` until scoring is complete.",
        "",
    ]
    key: dict[str, object] = {
        "facet_configuration": {"width": ANCHOR_WIDTH, "branch_count": ANCHOR_BRANCHES},
        "seeds": {},
    }
    for seed_id in PACKET_SEED_IDS:
        record = records_by_key[(seed_id, ANCHOR_WIDTH, ANCHOR_BRANCHES)]
        seed = record["seed"]
        lines.extend((f"## {seed['artist']} - {seed['title']}", ""))
        branch_order = sorted(
            range(ANCHOR_BRANCHES),
            key=lambda branch: stable_hash([EXPERIMENT_VERSION, seed_id, "branch", branch]),
        )
        branch_key: dict[str, int] = {}
        for display_number, branch in enumerate(branch_order):
            label = chr(ord("A") + display_number)
            branch_key[label] = branch + 1
            branch_record = record["branches"][branch]
            lines.append(f"### Direction {label}")
            for track in branch_record["display_previews"][:5]:
                lines.append(
                    f"- {track['artist'] or '?'} - {track['title'] or '?'} "
                    f"[`{track['track_id']}`]"
                )
            lines.append("")

        queue_names = sorted(
            ("raw_nearest", "mmr", "dpp"),
            key=lambda name: stable_hash([EXPERIMENT_VERSION, seed_id, "queue", name]),
        )
        queue_key: dict[str, str] = {}
        for display_number, name in enumerate(queue_names, start=1):
            label = str(display_number)
            queue_key[label] = name
            lines.append(f"### Flat queue {label}")
            for track in baseline_records[seed_id][name]["tracks"][:8]:
                lines.append(
                    f"- {track['artist'] or '?'} - {track['title'] or '?'} "
                    f"[`{track['track_id']}`]"
                )
            lines.append("")
        key["seeds"][str(seed_id)] = {
            "direction_display_to_embedding_branch": branch_key,
            "flat_queue_display_to_algorithm": queue_key,
        }
    atomic_text(output / "blind_packet.md", "\n".join(lines) + "\n")
    atomic_json(output / "blind_key.json", key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--audio-identity", type=Path, default=DEFAULT_AUDIO_IDENTITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    active, database_hash, catalog_hash = dedupe.load_active_library(
        args.db, args.active_catalog
    )
    quotient, identity_groups = quotient_proven_pcm_identities(active, args.audio_identity)
    by_id = {int(track_id): index for index, track_id in enumerate(quotient.track_ids)}
    missing = [seed_id for seed_id in SEED_IDS if seed_id not in by_id]
    if missing:
        raise ValueError(f"seed IDs absent from quotient domain: {missing}")

    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "evaluator_sha256": sha256_file(Path(__file__)),
        "database": str(args.db.resolve()),
        "database_sha256": database_hash,
        "active_catalog": str(args.active_catalog.resolve()),
        "active_catalog_sha256": catalog_hash,
        "audio_identity_evidence": str(args.audio_identity.resolve()),
        "audio_identity_evidence_sha256": sha256_file(args.audio_identity),
        "active_rows": active.count,
        "quotient_identities": quotient.count,
        "proven_pcm_identity_groups": identity_groups,
        "prior_facet_seed_ids": PRIOR_FACET_SEED_IDS,
        "extra_frozen_seed_ids": EXTRA_SEED_IDS,
        "neighborhood_widths": NEIGHBORHOOD_WIDTHS,
        "branch_counts": BRANCH_COUNTS,
        "anchor_configuration": {
            "width": ANCHOR_WIDTH,
            "branch_count": ANCHOR_BRANCHES,
        },
        "listening_gates": LISTENING_GATES,
        "intelligence_contract": (
            "Embeddings alone define retrieval, clustering, and scores. Decoded-PCM "
            "evidence defines the three identity quotients. Artist labels only audit "
            "and constrain displayed representatives after clustering."
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(args.output / "manifest.json", manifest)

    records: list[dict[str, object]] = []
    records_by_key: dict[tuple[int, int, int], dict[str, object]] = {}
    internals: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    baseline_records: dict[int, dict[str, dict[str, object]]] = {}
    baseline_internals: dict[int, dict[str, np.ndarray]] = {}

    for ordinal, seed_id in enumerate(SEED_IDS, start=1):
        seed_index = by_id[seed_id]
        context = matrix.seed_context(quotient, seed_index)
        baseline_records[seed_id], baseline_internals[seed_id] = baseline_queues(
            quotient, context
        )
        for width in NEIGHBORHOOD_WIDTHS:
            for branches in BRANCH_COUNTS:
                record, internal = cluster_record(
                    quotient, seed_index, context, width, branches
                )
                record["representation_comparison"] = representation_comparison(
                    quotient, internal, baseline_internals[seed_id]
                )
                key = (seed_id, width, branches)
                records.append(record)
                records_by_key[key] = record
                internals[key] = internal
        print(
            f"[{ordinal:02d}/{len(SEED_IDS)}] {quotient.artists[seed_index]} - "
            f"{quotient.titles[seed_index]}",
            flush=True,
        )

    stability = stability_results(quotient, internals)
    summary = summarize(records, stability)
    all_output_ids = {
        int(track["track_id"])
        for baseline in baseline_records.values()
        for algorithm in baseline.values()
        for track in algorithm["tracks"]
    }
    all_output_ids.update(
        int(track_id)
        for record in records
        for track_id in record["medoid_track_ids"]
    )
    leakage_groups = [
        group
        for group in identity_groups
        if sum(track_id in all_output_ids for track_id in group["member_track_ids"]) > 1
    ]
    decision = gate_decision(records, summary, len(leakage_groups))
    result = {
        "manifest": manifest,
        "records": records,
        "baselines": {str(seed_id): value for seed_id, value in baseline_records.items()},
        "stability": stability,
        "summary": summary,
        "exact_copy_leakage_groups": leakage_groups,
        "decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    result["result_payload_sha256"] = stable_hash(result)
    atomic_json(args.output / "results.json", result)
    write_blind_packet(args.output, quotient, records_by_key, baseline_records)
    print(f"Decision: {decision['decision']}", flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
