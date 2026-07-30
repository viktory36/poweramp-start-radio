#!/usr/bin/env python3
"""Full-library duplicate, version-preservation, and candidate-reach audit.

This is evidence code, not app code. Embeddings alone drive every ranking and
selector. Metadata is used only to label rows for inspection. Duplicate policy
is evaluated against independently decoded-audio evidence; metadata and cosine
thresholds are deliberately treated as diagnostic hypotheses, never truth.
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
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_device_feature_acceptance as phone_oracle
import v2_queue_eval as queue_eval
import v2_seed_conditioning_eval as seed_eval
import v2_selection_knob_matrix as matrix


REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATABASE = queue_eval.DEFAULT_DB
DEFAULT_ACTIVE_CATALOG = matrix.DEFAULT_ACTIVE_CATALOG
DEFAULT_TEXT_REPORT = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-realistic-text-battery"
    / "report.json"
)
DEFAULT_AUDIO_IDENTITY = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "recording-identity-policy-2026-07-15"
    / "results.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "dedupe-pool-2026-07-15"
)

EXPERIMENT_VERSION = "dedupe-pool-active-domain-v1"
QUEUE_SIZE = 30
POOL_REACHES = (0.02, 0.05, 1.0)
MMR_LAMBDAS = (0.4, matrix.DEFAULT_LAMBDA)
DPP_EXPONENTS = (1.0,)
THRESHOLDS = (0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9999)
TEXT_GREEDY_THRESHOLDS = (0.97, 0.99, 0.9999)
TEXT_INSPECTION_QUERIES = {
    "ambient",
    "sleep",
    "slow",
    "psychedelic",
    "slow psychedelic",
    "guitar",
    "relaxing",
    "rainy night",
    "Indian psychedelic electronic",
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


def load_active_library(
    database: Path,
    active_catalog: Path,
) -> tuple[queue_eval.Library, str, str]:
    database_hash = sha256_file(database)
    catalog_hash = sha256_file(active_catalog)
    if database_hash != queue_eval.EXPECTED_DB_SHA256:
        raise ValueError("frozen database hash mismatch")
    if catalog_hash != matrix.EXPECTED_CATALOG_SHA256:
        raise ValueError("active catalog hash mismatch")
    active_ids = phone_oracle.active_track_ids(active_catalog)
    if len(active_ids) != matrix.EXPECTED_ACTIVE_TRACKS:
        raise ValueError("active catalog count mismatch")
    full, loaded_hash = queue_eval.load_library(database)
    if loaded_hash != database_hash:
        raise ValueError("database changed while loading")
    positions = np.fromiter(
        (
            index
            for index, track_id in enumerate(full.track_ids)
            if int(track_id) in active_ids
        ),
        dtype=np.int64,
        count=len(active_ids),
    )
    if positions.size != matrix.EXPECTED_ACTIVE_TRACKS:
        raise ValueError("database/catalog intersection mismatch")
    return matrix.subset_library(full, positions), database_hash, catalog_hash


def row_label(library: queue_eval.Library, index: int) -> dict[str, object]:
    return {
        "track_id": int(library.track_ids[index]),
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "duration_ms": int(library.durations_ms[index]),
        "file_path": library.file_paths[index],
    }


def queue_ids(library: queue_eval.Library, selected: Sequence[int]) -> list[int]:
    return [int(library.track_ids[index]) for index in selected]


def queue_change(left: Sequence[int], right: Sequence[int]) -> dict[str, object]:
    left_ids = list(left)
    right_ids = list(right)
    intersection = len(set(left_ids) & set(right_ids))
    union = len(set(left_ids) | set(right_ids))
    return {
        "exact_order_equal": left_ids == right_ids,
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
        "same_position": sum(a == b for a, b in zip(left_ids, right_ids)),
    }


def queue_geometry(
    library: queue_eval.Library,
    context: matrix.SeedContext,
    selected: Sequence[int],
) -> dict[str, float | int | None]:
    if not selected:
        return {
            "returned": 0,
            "mean_seed_cosine": None,
            "mean_pairwise_cosine": None,
            "median_seed_rank": None,
        }
    indices = np.asarray(selected, dtype=np.int64)
    embeddings = library.embeddings[indices]
    pairwise = None
    if indices.size > 1:
        gram = embeddings @ embeddings.T
        pairwise = float(np.mean(gram[np.triu_indices(indices.size, 1)]))
    return {
        "returned": len(selected),
        "mean_seed_cosine": float(np.mean(context.seed_similarities[indices])),
        "mean_pairwise_cosine": pairwise,
        "median_seed_rank": float(np.median(context.seed_rank[indices])),
    }


def exact_embedding_groups(
    library: queue_eval.Library,
) -> tuple[dict[bytes, list[int]], dict[int, bytes]]:
    groups: dict[bytes, list[int]] = defaultdict(list)
    token_by_index: dict[int, bytes] = {}
    for index, embedding in enumerate(library.embeddings):
        token = hashlib.sha256(embedding.tobytes()).digest()
        groups[token].append(index)
        token_by_index[index] = token
    return groups, token_by_index


def summarize_exact_embedding_groups(
    library: queue_eval.Library,
    groups: dict[bytes, list[int]],
) -> dict[str, object]:
    repeated = [indices for indices in groups.values() if len(indices) > 1]
    repeated.sort(key=lambda values: (-len(values), int(library.track_ids[values[0]])))
    return {
        "group_count": len(repeated),
        "row_count": sum(len(values) for values in repeated),
        "excess_row_count": sum(len(values) - 1 for values in repeated),
        "maximum_group_size": max((len(values) for values in repeated), default=0),
        "examples": [
            {
                "size": len(indices),
                "rows": [row_label(library, index) for index in indices[:12]],
                "truncated": len(indices) > 12,
            }
            for indices in repeated[:12]
        ],
        "warning": (
            "Bit-identical model outputs are an audit proxy, not decoded-audio equality proof."
        ),
    }


def load_audio_pairs(
    path: Path,
    index_by_id: dict[int, int],
    library: queue_eval.Library,
) -> tuple[list[dict[str, object]], list[tuple[int, int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pairs: list[dict[str, object]] = []
    proven_pairs: list[tuple[int, int]] = []
    for record in raw["records"]:
        if record["source"] != "real_library":
            continue
        if record["evidence"].get("policy_scored") is False:
            continue
        left_id = int(record["left"]["track_id"])
        right_id = int(record["right"]["track_id"])
        if left_id not in index_by_id or right_id not in index_by_id:
            continue
        left = index_by_id[left_id]
        right = index_by_id[right_id]
        cosine = float(library.embeddings[left] @ library.embeddings[right])
        decoded_equal = bool(record["evidence"]["decoded_pcm_identical"])
        if decoded_equal:
            proven_pairs.append((left, right))
        pairs.append(
            {
                "name": record["name"],
                "expectation": record["expectation"],
                "subtype": record["subtype"],
                "decoded_pcm_identical": decoded_equal,
                "embedding_cosine": cosine,
                "left": row_label(library, left),
                "right": row_label(library, right),
            }
        )
    return pairs, proven_pairs


def threshold_confusion(pairs: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    binary = [pair for pair in pairs if pair["expectation"] != "ambiguous"]
    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        positive = [pair for pair in binary if pair["expectation"] == "same_rendition"]
        negative = [pair for pair in binary if pair["expectation"] == "distinct"]
        hits = [pair for pair in positive if float(pair["embedding_cosine"]) >= threshold]
        false_merges = [
            pair for pair in negative if float(pair["embedding_cosine"]) >= threshold
        ]
        ambiguous_merges = [
            pair
            for pair in pairs
            if pair["expectation"] == "ambiguous"
            and float(pair["embedding_cosine"]) >= threshold
        ]
        rows.append(
            {
                "threshold": threshold,
                "same_rendition_hits": len(hits),
                "same_rendition_total": len(positive),
                "misses": [pair["name"] for pair in positive if pair not in hits],
                "distinct_false_merges": [pair["name"] for pair in false_merges],
                "distinct_total": len(negative),
                "ambiguous_merges": [pair["name"] for pair in ambiguous_merges],
            }
        )
    return rows


def candidate_subset_without(
    candidates: np.ndarray,
    relevance: np.ndarray,
    excluded: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.fromiter(
        (int(index) not in excluded for index in candidates),
        dtype=np.bool_,
        count=candidates.size,
    )
    return candidates[keep], relevance[keep]


def select(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    config: matrix.SelectorConfig,
) -> list[int]:
    if config.mode == "mmr":
        return matrix.select_mmr(library, candidates, relevance, config)[0]
    if config.mode == "dpp":
        return matrix.select_dpp(library, candidates, relevance, config)[0]
    raise ValueError(config.mode)


def seed_duplicate_and_pool_cases(
    library: queue_eval.Library,
    proven_pairs: Sequence[tuple[int, int]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    duplicate_cases: list[dict[str, object]] = []
    pool_cases: list[dict[str, object]] = []

    for left, right in proven_pairs:
        for seed, partner in ((left, right), (right, left)):
            context = matrix.seed_context(library, seed)
            pool_count = matrix.effective_pool_count(
                library.count - 1,
                matrix.SelectorConfig(mode="mmr", reach=0.02),
            )
            candidates = context.closest_order[:pool_count]
            relevance = context.seed_similarities[candidates]
            gram = library.embeddings[candidates] @ library.embeddings[candidates].T
            partner_locations = np.flatnonzero(candidates == partner)
            partner_rank = (
                int(partner_locations[0]) + 1 if partner_locations.size else None
            )

            for mode, control in (("mmr", matrix.DEFAULT_LAMBDA), ("dpp", 1.0)):
                config = matrix.SelectorConfig(
                    mode=mode,
                    queue_size=QUEUE_SIZE,
                    reach=0.02,
                    mmr_lambda=control if mode == "mmr" else matrix.DEFAULT_LAMBDA,
                    dpp_exponent=control if mode == "dpp" else 1.0,
                )
                baseline = select(library, candidates, relevance, config)
                filtered_candidates, filtered_relevance = candidate_subset_without(
                    candidates, relevance, {partner}
                )
                exact_exclusion = select(
                    library, filtered_candidates, filtered_relevance, config
                )
                if mode == "mmr":
                    seed_conditioned = list(
                        seed_eval.select_mmr(
                            library=library,
                            seed_index=seed,
                            candidates=candidates,
                            relevance=relevance,
                            gram=gram,
                            count=QUEUE_SIZE,
                            lambda_=control,
                            include_seed_in_history=True,
                        ).selected
                    )
                else:
                    seed_conditioned = list(
                        seed_eval.select_dpp(
                            library=library,
                            seed_index=seed,
                            candidates=candidates,
                            relevance=relevance,
                            gram=gram,
                            count=QUEUE_SIZE,
                            quality_exponent=control,
                            condition_on_seed=True,
                        ).selected
                    )
                duplicate_cases.append(
                    {
                        "seed": row_label(library, seed),
                        "proven_duplicate": row_label(library, partner),
                        "selector": mode,
                        "control": control,
                        "candidate_count": int(candidates.size),
                        "duplicate_seed_rank": partner_rank,
                        "baseline_duplicate_queue_position": (
                            baseline.index(partner) + 1 if partner in baseline else None
                        ),
                        "baseline_track_ids": queue_ids(library, baseline),
                        "exact_pcm_exclusion_track_ids": queue_ids(
                            library, exact_exclusion
                        ),
                        "seed_conditioned_track_ids": queue_ids(
                            library, seed_conditioned
                        ),
                        "seed_conditioned_duplicate_queue_position": (
                            seed_conditioned.index(partner) + 1
                            if partner in seed_conditioned
                            else None
                        ),
                        "change_after_exact_exclusion": queue_change(
                            queue_ids(library, baseline),
                            queue_ids(library, exact_exclusion),
                        ),
                        "change_after_seed_conditioning": queue_change(
                            queue_ids(library, baseline),
                            queue_ids(library, seed_conditioned),
                        ),
                    }
                )

        # One orientation per proven pair is enough for the reach experiment.
        seed, partner = left, right
        context = matrix.seed_context(library, seed)
        for mode, controls in (("mmr", MMR_LAMBDAS), ("dpp", DPP_EXPONENTS)):
            for control in controls:
                queues: dict[str, list[int]] = {}
                geometry: dict[str, dict[str, float | int | None]] = {}
                for reach in POOL_REACHES:
                    config = matrix.SelectorConfig(
                        mode=mode,
                        queue_size=QUEUE_SIZE,
                        reach=reach,
                        mmr_lambda=(control if mode == "mmr" else matrix.DEFAULT_LAMBDA),
                        dpp_exponent=(control if mode == "dpp" else 1.0),
                    )
                    candidates, relevance, _ = matrix.candidates_for(
                        library, context, config
                    )
                    candidates, relevance = candidate_subset_without(
                        candidates, relevance, {partner}
                    )
                    selected = select(library, candidates, relevance, config)
                    key = f"{reach:.2f}"
                    queues[key] = queue_ids(library, selected)
                    geometry[key] = queue_geometry(library, context, selected)
                full = queues["1.00"]
                pool_cases.append(
                    {
                        "seed": row_label(library, seed),
                        "excluded_proven_duplicate": row_label(library, partner),
                        "selector": mode,
                        "control": control,
                        "queues": queues,
                        "geometry": geometry,
                        "two_percent_vs_full": queue_change(queues["0.02"], full),
                        "five_percent_vs_full": queue_change(queues["0.05"], full),
                    }
                )
    return duplicate_cases, pool_cases


def load_text_queries(path: Path) -> dict[str, np.ndarray]:
    report = json.loads(path.read_text(encoding="utf-8"))
    queries: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for run in report["textRuns"]:
        query = str(run["query"])
        embedding = np.asarray(run["embedding"], dtype=np.float32)
        if embedding.size != queue_eval.EXPECTED_DIM:
            raise ValueError(f"text embedding {query!r} has wrong dimension")
        digest = hashlib.sha256(embedding.astype("<f4", copy=False).tobytes()).hexdigest()
        previous = hashes.setdefault(query, digest)
        if previous != digest:
            raise ValueError(f"repeated phone text query changed embedding: {query}")
        queries.setdefault(query, embedding)
    return queries


def exact_pcm_tokens(
    library: queue_eval.Library,
    proven_pairs: Sequence[tuple[int, int]],
) -> dict[int, str]:
    tokens: dict[int, str] = {}
    for ordinal, (left, right) in enumerate(proven_pairs):
        token = f"curated-exact-pcm-{ordinal}"
        tokens[left] = token
        tokens[right] = token
    return tokens


def proxy_excess(tokens: Iterable[str]) -> int:
    seen: set[str] = set()
    excess = 0
    for token in tokens:
        if token in seen:
            excess += 1
        else:
            seen.add(token)
    return excess


def normalized_artist_title(library: queue_eval.Library, index: int) -> str:
    artist = matrix.normalized_text(library.artists[index])
    title = matrix.normalized_text(library.titles[index])
    return f"{artist}|{title}" if artist or title else f"unknown:{index}"


def greedy_similarity_suppression(
    library: queue_eval.Library,
    order: np.ndarray,
    threshold: float,
    count: int,
) -> tuple[list[int], list[dict[str, object]], int]:
    selected: list[int] = []
    rejected: list[dict[str, object]] = []
    scanned = 0
    for raw_index in order:
        index = int(raw_index)
        scanned += 1
        blocker = None
        if selected:
            similarities = library.embeddings[np.asarray(selected)] @ library.embeddings[index]
            best = int(np.argmax(similarities))
            if float(similarities[best]) >= threshold:
                blocker = selected[best]
                rejected.append(
                    {
                        "candidate": row_label(library, index),
                        "blocked_by": row_label(library, blocker),
                        "audio_embedding_cosine": float(similarities[best]),
                    }
                )
        if blocker is None:
            selected.append(index)
            if len(selected) == count:
                break
    return selected, rejected, scanned


def exact_pcm_reduce(
    order: np.ndarray,
    tokens: dict[int, str],
    count: int,
) -> tuple[list[int], int, int]:
    selected: list[int] = []
    seen: set[str] = set()
    collapsed = 0
    scanned = 0
    for raw_index in order:
        index = int(raw_index)
        scanned += 1
        token = tokens.get(index)
        if token is not None and token in seen:
            collapsed += 1
            continue
        if token is not None:
            seen.add(token)
        selected.append(index)
        if len(selected) == count:
            break
    return selected, collapsed, scanned


def text_cases(
    library: queue_eval.Library,
    queries: dict[str, np.ndarray],
    embedding_token_by_index: dict[int, bytes],
    proven_pairs: Sequence[tuple[int, int]],
    audio_pairs: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    index_by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    pcm_tokens = exact_pcm_tokens(library, proven_pairs)
    results: list[dict[str, object]] = []
    for query, embedding in queries.items():
        similarities = (library.embeddings @ embedding).astype(np.float32, copy=False)
        order = np.lexsort((library.track_ids, -similarities)).astype(np.int64, copy=False)
        baseline = [int(index) for index in order[:QUEUE_SIZE]]
        top100 = [int(index) for index in order[:100]]
        exact_pcm, collapsed, exact_scanned = exact_pcm_reduce(
            order, pcm_tokens, QUEUE_SIZE
        )
        near_policies: list[dict[str, object]] = []
        for threshold in TEXT_GREEDY_THRESHOLDS:
            selected, rejected, scanned = greedy_similarity_suppression(
                library, order, threshold, QUEUE_SIZE
            )
            near_policies.append(
                {
                    "threshold": threshold,
                    "scanned_rows": scanned,
                    "rejected_rows": len(rejected),
                    "mean_query_cosine": float(np.mean(similarities[selected])),
                    "baseline_change": queue_change(
                        queue_ids(library, baseline), queue_ids(library, selected)
                    ),
                    "selected_track_ids": queue_ids(library, selected),
                    "rejection_examples": rejected[:8],
                }
            )

        pair_ranks: list[dict[str, object]] = []
        inverse_rank = np.empty(library.count, dtype=np.int32)
        inverse_rank[order] = np.arange(1, library.count + 1, dtype=np.int32)
        for pair in audio_pairs:
            left_id = int(pair["left"]["track_id"])
            right_id = int(pair["right"]["track_id"])
            left = index_by_id[left_id]
            right = index_by_id[right_id]
            left_rank = int(inverse_rank[left])
            right_rank = int(inverse_rank[right])
            if min(left_rank, right_rank) <= 100 or query in TEXT_INSPECTION_QUERIES:
                pair_ranks.append(
                    {
                        "name": pair["name"],
                        "expectation": pair["expectation"],
                        "subtype": pair["subtype"],
                        "embedding_cosine": pair["embedding_cosine"],
                        "left_rank": left_rank,
                        "right_rank": right_rank,
                    }
                )

        results.append(
            {
                "query": query,
                "baseline_track_ids": queue_ids(library, baseline),
                "baseline_mean_query_cosine": float(np.mean(similarities[baseline])),
                "baseline_exact_embedding_excess": proxy_excess(
                    embedding_token_by_index[index].hex() for index in baseline
                ),
                "baseline_artist_title_proxy_excess": proxy_excess(
                    normalized_artist_title(library, index) for index in baseline
                ),
                "top100_exact_embedding_excess": proxy_excess(
                    embedding_token_by_index[index].hex() for index in top100
                ),
                "top100_artist_title_proxy_excess": proxy_excess(
                    normalized_artist_title(library, index) for index in top100
                ),
                "curated_exact_pcm_reduction": {
                    "collapsed": collapsed,
                    "scanned_rows": exact_scanned,
                    "track_ids": queue_ids(library, exact_pcm),
                    "baseline_change": queue_change(
                        queue_ids(library, baseline), queue_ids(library, exact_pcm)
                    ),
                },
                "near_similarity_policies": near_policies,
                "curated_pair_ranks": pair_ranks,
                "top30": [
                    {
                        "rank": rank,
                        "query_cosine": float(similarities[index]),
                        **row_label(library, index),
                    }
                    for rank, index in enumerate(baseline, start=1)
                ],
            }
        )
    return results


def numeric_summary(values: Iterable[float]) -> dict[str, float | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def summary(
    exact_groups: dict[str, object],
    thresholds: Sequence[dict[str, object]],
    duplicate_cases: Sequence[dict[str, object]],
    pool_cases: Sequence[dict[str, object]],
    text_results: Sequence[dict[str, object]],
) -> dict[str, object]:
    by_pool: dict[str, object] = {}
    for selector in ("mmr", "dpp"):
        for control in sorted(
            {float(case["control"]) for case in pool_cases if case["selector"] == selector}
        ):
            cases = [
                case
                for case in pool_cases
                if case["selector"] == selector and float(case["control"]) == control
            ]
            by_pool[f"{selector}:{control:.8g}"] = {
                "case_count": len(cases),
                "two_percent_exact_full_matches": sum(
                    bool(case["two_percent_vs_full"]["exact_order_equal"])
                    for case in cases
                ),
                "five_percent_exact_full_matches": sum(
                    bool(case["five_percent_vs_full"]["exact_order_equal"])
                    for case in cases
                ),
                "two_percent_full_jaccard": numeric_summary(
                    float(case["two_percent_vs_full"]["jaccard"]) for case in cases
                ),
                "five_percent_full_jaccard": numeric_summary(
                    float(case["five_percent_vs_full"]["jaccard"]) for case in cases
                ),
            }
    return {
        "exact_embedding_proxy": exact_groups,
        "cosine_threshold_confusion": list(thresholds),
        "seed_duplicate_replay": {
            "cases": len(duplicate_cases),
            "proven_duplicate_returned": sum(
                case["baseline_duplicate_queue_position"] is not None
                for case in duplicate_cases
            ),
            "duplicate_queue_position": numeric_summary(
                float(case["baseline_duplicate_queue_position"])
                for case in duplicate_cases
                if case["baseline_duplicate_queue_position"] is not None
            ),
            "mean_jaccard_after_exact_exclusion": float(
                np.mean(
                    [
                        float(case["change_after_exact_exclusion"]["jaccard"])
                        for case in duplicate_cases
                    ]
                )
            ),
            "seed_conditioning_removed_proven_duplicate": sum(
                case["seed_conditioned_duplicate_queue_position"] is None
                for case in duplicate_cases
            ),
            "mean_jaccard_after_seed_conditioning": float(
                np.mean(
                    [
                        float(case["change_after_seed_conditioning"]["jaccard"])
                        for case in duplicate_cases
                    ]
                )
            ),
        },
        "focused_pool": by_pool,
        "text": {
            "query_count": len(text_results),
            "top30_exact_embedding_excess": sum(
                int(result["baseline_exact_embedding_excess"])
                for result in text_results
            ),
            "top30_artist_title_proxy_excess": sum(
                int(result["baseline_artist_title_proxy_excess"])
                for result in text_results
            ),
            "queries_with_curated_exact_pcm_collapse": sum(
                int(result["curated_exact_pcm_reduction"]["collapsed"]) > 0
                for result in text_results
            ),
            "near_policy": {
                f"{threshold:.4f}": {
                    "mean_baseline_jaccard": float(
                        np.mean(
                            [
                                next(
                                    policy
                                    for policy in result["near_similarity_policies"]
                                    if float(policy["threshold"]) == threshold
                                )["baseline_change"]["jaccard"]
                                for result in text_results
                            ]
                        )
                    ),
                    "mean_rejected_rows": float(
                        np.mean(
                            [
                                next(
                                    policy
                                    for policy in result["near_similarity_policies"]
                                    if float(policy["threshold"]) == threshold
                                )["rejected_rows"]
                                for result in text_results
                            ]
                        )
                    ),
                }
                for threshold in TEXT_GREEDY_THRESHOLDS
            },
        },
    }


def render_qualitative(results: dict[str, object]) -> str:
    lines = [
        "# Dedupe And Candidate-Pool Qualitative Packet",
        "",
        "Metadata below labels embedding-only results; it never changes ranking.",
        "",
        "## Curated audio pairs",
        "",
        "| Expectation | Type | Cosine | Pair |",
        "| --- | --- | ---: | --- |",
    ]
    for pair in results["audio_pairs"]:
        lines.append(
            f"| {pair['expectation']} | {pair['subtype']} | "
            f"{float(pair['embedding_cosine']):.6f} | "
            f"{pair['left']['artist']} - {pair['left']['title']} "
            f"({pair['left']['track_id']}) / {pair['right']['track_id']} |"
        )
    lines.extend(["", "## Seed duplicate replay", ""])
    for case in results["seed_duplicate_cases"]:
        lines.append(
            f"- {case['selector'].upper()} from {case['seed']['artist']} - "
            f"{case['seed']['title']} ({case['seed']['track_id']}): proven copy "
            f"{case['proven_duplicate']['track_id']} was queue position "
            f"{case['baseline_duplicate_queue_position']}; exact exclusion Jaccard "
            f"{float(case['change_after_exact_exclusion']['jaccard']):.3f}; treating the "
            f"seed as selected left the copy at "
            f"{case['seed_conditioned_duplicate_queue_position']} and changed-set Jaccard "
            f"{float(case['change_after_seed_conditioning']['jaccard']):.3f}."
        )
    lines.extend(["", "## Text queries with version/copy evidence", ""])
    for result in results["text_cases"]:
        if result["query"] not in TEXT_INSPECTION_QUERIES:
            continue
        lines.append(f"### {result['query']}")
        lines.append("")
        lines.append(
            f"Top-30 exact-vector excess: {result['baseline_exact_embedding_excess']}; "
            f"artist/title proxy excess: {result['baseline_artist_title_proxy_excess']}; "
            f"curated exact-PCM collapses: "
            f"{result['curated_exact_pcm_reduction']['collapsed']}."
        )
        lines.append("")
        for row in result["top30"][:12]:
            lines.append(
                f"{row['rank']}. {row['artist']} - {row['title']} "
                f"[{row['track_id']}] cosine={float(row['query_cosine']):.5f}"
            )
        relevant_pairs = [
            pair
            for pair in result["curated_pair_ranks"]
            if min(int(pair["left_rank"]), int(pair["right_rank"])) <= 100
        ]
        if relevant_pairs:
            lines.extend(["", "Curated pair ranks:"])
            for pair in relevant_pairs:
                lines.append(
                    f"- {pair['name']}: {pair['left_rank']} / {pair['right_rank']} "
                    f"({pair['expectation']}, cosine "
                    f"{float(pair['embedding_cosine']):.5f})"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--text-report", type=Path, default=DEFAULT_TEXT_REPORT)
    parser.add_argument("--audio-identity", type=Path, default=DEFAULT_AUDIO_IDENTITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    library, database_hash, catalog_hash = load_active_library(
        args.database, args.active_catalog
    )
    index_by_id = {
        int(track_id): index for index, track_id in enumerate(library.track_ids)
    }
    groups, embedding_tokens = exact_embedding_groups(library)
    exact_groups = summarize_exact_embedding_groups(library, groups)
    audio_pairs, proven_pairs = load_audio_pairs(
        args.audio_identity, index_by_id, library
    )
    threshold_rows = threshold_confusion(audio_pairs)
    duplicate_cases, pool_cases = seed_duplicate_and_pool_cases(library, proven_pairs)
    queries = load_text_queries(args.text_report)
    text_results = text_cases(
        library, queries, embedding_tokens, proven_pairs, audio_pairs
    )
    result_summary = summary(
        exact_groups, threshold_rows, duplicate_cases, pool_cases, text_results
    )
    payload: dict[str, object] = {
        "experiment_version": EXPERIMENT_VERSION,
        "inputs": {
            "database": str(args.database.resolve()),
            "database_sha256": database_hash,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": catalog_hash,
            "active_track_count": library.count,
            "text_report": str(args.text_report.resolve()),
            "text_report_sha256": sha256_file(args.text_report),
            "audio_identity": str(args.audio_identity.resolve()),
            "audio_identity_sha256": sha256_file(args.audio_identity),
        },
        "contract": {
            "ranking_inputs": "CLaMP3 audio/text embeddings only",
            "metadata_role": "labels only",
            "duplicate_ground_truth": (
                "exact complete decoded PCM equality at native sample rate/channel layout"
            ),
            "determinism": "numeric score then active track ID",
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "audio_pairs": audio_pairs,
        "cosine_threshold_confusion": threshold_rows,
        "seed_duplicate_cases": duplicate_cases,
        "pool_cases": pool_cases,
        "text_cases": text_results,
        "summary": result_summary,
    }
    payload["deterministic_payload_sha256"] = stable_hash(payload)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "results.json", payload)
    atomic_json(args.output / "summary.json", result_summary)
    atomic_text(args.output / "qualitative.md", render_qualitative(payload))
    manifest = {
        "experiment_version": EXPERIMENT_VERSION,
        "elapsed_seconds": time.perf_counter() - started,
        "results_sha256": sha256_file(args.output / "results.json"),
        "summary_sha256": sha256_file(args.output / "summary.json"),
        "qualitative_sha256": sha256_file(args.output / "qualitative.md"),
        "script_sha256": sha256_file(Path(__file__)),
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"summary": result_summary, "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
