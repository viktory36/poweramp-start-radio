#!/usr/bin/env python3
"""Evaluate exact-angle and neutral-residual song edits on the active phone library.

This is a host-only retrieval experiment. Embeddings are the only ranking input;
track metadata is emitted solely for qualitative labels and duplicate diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import v2_active_composition_eval as composition


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_NEUTRAL_RESULTS = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "residual-direction-neutral-prompts-2026-07-15"
    / "prompt-results.jsonl"
)
DEFAULT_TEXT_RESULTS = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text"
    / "prompt-results.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "residual-direction-2026-07-15"
)

# The first five values are the requested sweep. Sixty and seventy-five degrees
# probe the remaining valid arc (the smallest signed seed/target arc is ~79 deg)
# so a late transition cannot be mistaken for a total failure.
ANGLES_DEG: tuple[int, ...] = (5, 10, 20, 30, 45, 60, 75)
TOP_K = 30
DIAGNOSTIC_K = 100
NEAR_DUPLICATE_COSINE = 0.995
REFINE_WIDTHS: tuple[float, ...] = (0.0025, 0.005, 0.01)


# Song IDs and weights preserve the already measured composition cases. These definitions
# were frozen before this evaluator looked at any residual-direction ranking.
CASES: tuple[dict[str, object], ...] = (
    {
        "id": "bonobo_with_ambient",
        "seed_track_id": 80437,
        "text": "ambient",
        "sign": 1,
        "seed_weight": 0.65,
        "text_weight": 0.35,
    },
    {
        "id": "neroche_with_sleep",
        "seed_track_id": 42335,
        "text": "sleep",
        "sign": 1,
        "seed_weight": 0.65,
        "text_weight": 0.35,
    },
    {
        "id": "khruangbin_with_desert_blues",
        "seed_track_id": 5987,
        "text": "desert blues",
        "sign": 1,
        "seed_weight": 0.65,
        "text_weight": 0.35,
    },
    {
        "id": "hallucinogen_with_dark_ambient",
        "seed_track_id": 13384,
        "text": "dark ambient",
        "sign": 1,
        "seed_weight": 0.65,
        "text_weight": 0.35,
    },
    {
        "id": "nusrat_with_fast_broken_beat",
        "seed_track_id": 209,
        "text": "fast broken beat",
        "sign": 1,
        "seed_weight": 0.65,
        "text_weight": 0.35,
    },
    {
        "id": "tool_with_intricate_acoustic_percussion",
        "seed_track_id": 31830,
        "text": "intricate acoustic percussion",
        "sign": 1,
        "seed_weight": 0.70,
        "text_weight": 0.30,
    },
    {
        "id": "tool_away_distorted_electric_guitar",
        "seed_track_id": 31830,
        "text": "distorted electric guitar",
        "sign": -1,
        "seed_weight": 0.80,
        "text_weight": 0.20,
    },
    {
        "id": "kasabian_away_guitar",
        "seed_track_id": 38327,
        "text": "guitar",
        "sign": -1,
        "seed_weight": 0.75,
        "text_weight": 0.25,
    },
)


def sha256_vector(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return (value / norm).astype(np.float32)


def load_prompt_jsonl(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    embeddings: dict[str, np.ndarray] = {}
    provenance: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = str(row["prompt"])
        vector = unit(np.asarray(row["embedding"], dtype=np.float32))
        if vector.shape != (composition.queue_eval.EXPECTED_DIM,):
            raise ValueError(f"{path}:{line_number} has the wrong embedding dimension")
        if prompt in embeddings:
            raise ValueError(f"duplicate prompt {prompt!r} in {path}")
        embeddings[prompt] = vector
        provenance[prompt] = {
            "source": str(path),
            "source_sha256": composition.sha256_file(path),
            "row_id": row.get("id"),
            "kind": row.get("kind"),
            "embedding_sha256": sha256_vector(vector),
        }
    return embeddings, provenance


def signed_text_percentiles(
    cache: composition.AnchorCache,
    text_reference: str,
    sign: int,
) -> np.ndarray:
    return cache.percentiles(text_reference, negative=sign < 0)


def exact_angle_direction(
    seed: np.ndarray,
    target: np.ndarray,
    angle_degrees: float,
) -> tuple[np.ndarray, float]:
    """Move exactly angle_degrees from seed on the seed/target great circle."""
    seed64 = unit(seed).astype(np.float64)
    target64 = unit(target).astype(np.float64)
    dot = float(np.clip(np.dot(seed64, target64), -1.0, 1.0))
    available = math.acos(dot)
    requested = math.radians(angle_degrees)
    if requested > available + 1e-12:
        raise ValueError(
            f"requested {angle_degrees:g} degrees exceeds target arc "
            f"{math.degrees(available):.6f} degrees"
        )
    tangent = target64 - dot * seed64
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("seed and target do not define a great-circle direction")
    tangent /= tangent_norm
    query = math.cos(requested) * seed64 + math.sin(requested) * tangent
    return unit(query), math.degrees(available)


def neutral_residual_direction(
    seed: np.ndarray,
    text: np.ndarray,
    neutral: np.ndarray,
    sign: int,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project text-neutral into the tangent plane at seed, then rotate exactly."""
    seed64 = unit(seed).astype(np.float64)
    difference = unit(text).astype(np.float64) - unit(neutral).astype(np.float64)
    tangent = difference - float(np.dot(difference, seed64)) * seed64
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("neutral residual has no seed-tangent component")
    tangent = sign * tangent / tangent_norm
    radians = math.radians(angle_degrees)
    query = math.cos(radians) * seed64 + math.sin(radians) * tangent
    return unit(query), tangent.astype(np.float32)


def exact_seed_angle_degrees(seed: np.ndarray, query: np.ndarray) -> float:
    # Re-normalize in Float64 for the diagnostic. A Float32 self-dot can be
    # 0.99999994, which falsely reports roughly 0.02 degrees at the endpoint.
    seed64 = np.asarray(seed, dtype=np.float64)
    query64 = np.asarray(query, dtype=np.float64)
    seed64 /= np.linalg.norm(seed64)
    query64 /= np.linalg.norm(query64)
    cosine = float(np.clip(np.dot(seed64, query64), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def overlap(first: Sequence[int], second: Sequence[int]) -> dict[str, float | int]:
    first_set = set(first)
    second_set = set(second)
    intersection = len(first_set & second_set)
    union = len(first_set | second_set)
    return {
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
        "same_position": sum(a == b for a, b in zip(first, second)),
    }


def normalized_label(library: composition.ActiveLibrary, index: int) -> tuple[str, str]:
    return (
        (library.artists[index] or "").strip().casefold(),
        (library.titles[index] or "").strip().casefold(),
    )


def duplicate_diagnostics(
    library: composition.ActiveLibrary,
    selected: np.ndarray,
) -> dict[str, object]:
    top = selected[:TOP_K]
    labels = [normalized_label(library, int(index)) for index in top]
    populated_labels = [label for label in labels if any(label)]
    exact_excess = len(populated_labels) - len(set(populated_labels))

    accepted: list[int] = []
    near_groups: list[dict[str, object]] = []
    for index in top:
        duplicate_of: int | None = None
        duplicate_cosine = -1.0
        for kept in accepted:
            cosine = float(np.dot(library.embeddings[int(index)], library.embeddings[kept]))
            if cosine >= NEAR_DUPLICATE_COSINE and cosine > duplicate_cosine:
                duplicate_of = kept
                duplicate_cosine = cosine
        if duplicate_of is None:
            accepted.append(int(index))
        else:
            near_groups.append(
                {
                    "track_id": int(library.track_ids[int(index)]),
                    "duplicate_of_track_id": int(library.track_ids[duplicate_of]),
                    "embedding_cosine": duplicate_cosine,
                    "same_normalized_artist_title": (
                        normalized_label(library, int(index))
                        == normalized_label(library, duplicate_of)
                    ),
                }
            )
    return {
        "exact_artist_title_excess": exact_excess,
        "embedding_ge_0_995_greedy_excess": len(near_groups),
        "embedding_ge_0_995_pairs": near_groups,
        "warning": "diagnostic only; neither metadata nor an embedding threshold is a safe identity rule",
    }


def duplicate_diagnostics_for_track_ids(
    library: composition.ActiveLibrary,
    track_ids: Sequence[int],
) -> dict[str, object]:
    positions = np.searchsorted(library.track_ids, np.asarray(track_ids, dtype=np.int64))
    if np.any(positions >= library.count) or not np.array_equal(
        library.track_ids[positions], np.asarray(track_ids, dtype=np.int64)
    ):
        raise AssertionError("duplicate diagnostics received a non-active track ID")
    return duplicate_diagnostics(library, positions)


def selected_diagnostics(
    library: composition.ActiveLibrary,
    selected: np.ndarray,
    seed_percentiles: np.ndarray,
    text_percentiles: np.ndarray,
) -> dict[str, object]:
    top = selected[:TOP_K]
    seed_values = seed_percentiles[top]
    text_values = text_percentiles[top]
    row_worst = np.minimum(seed_values, text_values)
    threshold = 1.0 - 100.0 / library.count
    return {
        "mean_seed_percentile": float(np.mean(seed_values)),
        "minimum_seed_percentile": float(np.min(seed_values)),
        "mean_effective_text_percentile": float(np.mean(text_values)),
        "minimum_effective_text_percentile": float(np.min(text_values)),
        "mean_row_worst_percentile": float(np.mean(row_worst)),
        "minimum_row_worst_percentile": float(np.min(row_worst)),
        "seed_top100_coverage": int(np.sum(seed_values > threshold)),
        "effective_text_top100_coverage": int(np.sum(text_values > threshold)),
        "distinct_display_artists": len(
            {
                (library.artists[int(index)] or "").strip().casefold()
                for index in top
                if library.artists[int(index)]
            }
        ),
        "duplicates": duplicate_diagnostics(library, selected),
    }


def vector_ranking(
    library: composition.ActiveLibrary,
    cache: composition.AnchorCache,
    seed_reference: str,
    text_reference: str,
    sign: int,
    query: np.ndarray,
    method: str,
    angle_degrees: float | None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    seed = cache.embedding(seed_reference)
    text = cache.embedding(text_reference)
    seed_similarities = cache.similarities(seed_reference)
    text_similarities = cache.similarities(text_reference)
    seed_percentiles = cache.percentiles(seed_reference, negative=False)
    text_percentiles = signed_text_percentiles(cache, text_reference, sign)
    objective = np.asarray(library.embeddings @ unit(query), dtype=np.float32)
    excluded = {int(seed_reference.split(":", 1)[1])}
    selected = composition.rank_scalar(
        library.track_ids, objective, excluded, DIAGNOSTIC_K
    )
    repeat = composition.rank_scalar(
        library.track_ids, objective.copy(), excluded, DIAGNOSTIC_K
    )
    if not np.array_equal(selected, repeat):
        raise AssertionError(f"{method} was not deterministic")
    actual_angle = exact_seed_angle_degrees(seed, query)
    if angle_degrees is not None and abs(actual_angle - angle_degrees) > 2e-4:
        raise AssertionError(
            f"{method} requested {angle_degrees:g} degrees but produced {actual_angle:.8f}"
        )

    rows = [
        composition.row_record(
            library,
            int(index),
            rank,
            float(objective[index]),
            [seed_similarities, text_similarities],
            [seed_percentiles, text_percentiles],
        )
        for rank, index in enumerate(selected[:TOP_K], start=1)
    ]
    record: dict[str, object] = {
        "method": method,
        "requested_angle_degrees": angle_degrees,
        "actual_seed_angle_degrees": actual_angle,
        "query_vector_sha256": sha256_vector(query),
        "query_cosine_to_seed": float(np.dot(unit(query), seed)),
        "query_cosine_to_text": float(np.dot(unit(query), text)),
        "query_effective_text_cosine": float(sign * np.dot(unit(query), text)),
        "diagnostics": selected_diagnostics(
            library, selected, seed_percentiles, text_percentiles
        ),
        "top30": rows,
        "top100_track_ids": [int(library.track_ids[index]) for index in selected],
        "deterministic_repeat_exact": True,
    }
    if extra:
        record.update(extra)
    return record


def record_track_ids(record: dict[str, object]) -> list[int]:
    return [int(row["track_id"]) for row in record["top30"]]


def operator_track_ids(record: dict[str, object]) -> list[int]:
    return [int(row["track_id"]) for row in record["top30"]]


def refine_width_track_ids(record: dict[str, object]) -> list[int]:
    return [int(row["track_id"]) for row in record["top30"]]


def attach_baseline_overlaps(
    records: Iterable[dict[str, object]],
    baselines: dict[str, Sequence[int]],
) -> None:
    for record in records:
        ids = record_track_ids(record)
        record["top30_overlap_with_baselines"] = {
            label: overlap(ids, baseline) for label, baseline in baselines.items()
        }


def series_control_diagnostics(records: Sequence[dict[str, object]]) -> dict[str, object]:
    if [record["requested_angle_degrees"] for record in records] != list(ANGLES_DEG):
        raise AssertionError("control series has unexpected angles")

    def values(path: str) -> list[float]:
        if path.startswith("diagnostics."):
            key = path.split(".", 1)[1]
            return [float(record["diagnostics"][key]) for record in records]
        return [float(record[path]) for record in records]

    def violations(sequence: Sequence[float], direction: str) -> int:
        if direction == "up":
            return sum(second + 1e-9 < first for first, second in zip(sequence, sequence[1:]))
        return sum(second > first + 1e-9 for first, second in zip(sequence, sequence[1:]))

    query_effective = values("query_effective_text_cosine")
    selected_seed = values("diagnostics.mean_seed_percentile")
    selected_text = values("diagnostics.mean_effective_text_percentile")
    return {
        "query_effective_text_cosines": query_effective,
        "mean_selected_seed_percentiles": selected_seed,
        "mean_selected_effective_text_percentiles": selected_text,
        "query_effective_text_monotonic_increase_violations": violations(query_effective, "up"),
        "selected_seed_monotonic_decrease_violations": violations(selected_seed, "down"),
        "selected_effective_text_monotonic_increase_violations": violations(selected_text, "up"),
        "adjacent_top30_overlaps": [
            overlap(record_track_ids(first), record_track_ids(second))
            for first, second in zip(records, records[1:])
        ],
    }


def neutral_sensitivity(
    by_neutral: dict[str, Sequence[dict[str, object]]],
) -> list[dict[str, object]]:
    names = sorted(by_neutral)
    records: list[dict[str, object]] = []
    for angle_index, angle in enumerate(ANGLES_DEG):
        current = {name: by_neutral[name][angle_index] for name in names}
        pair_overlaps = [
            overlap(record_track_ids(current[first]), record_track_ids(current[second]))
            for first, second in itertools.combinations(names, 2)
        ]
        vectors = {
            name: np.asarray(current[name]["query_vector"], dtype=np.float32)
            for name in names
        }
        pair_cosines = [
            float(np.dot(vectors[first], vectors[second]))
            for first, second in itertools.combinations(names, 2)
        ]
        sets = [set(record_track_ids(record)) for record in current.values()]
        seed_means = [
            float(record["diagnostics"]["mean_seed_percentile"])
            for record in current.values()
        ]
        text_means = [
            float(record["diagnostics"]["mean_effective_text_percentile"])
            for record in current.values()
        ]
        records.append(
            {
                "angle_degrees": angle,
                "neutral_variants": names,
                "pairwise_query_cosine_mean": float(np.mean(pair_cosines)),
                "pairwise_query_cosine_minimum": float(np.min(pair_cosines)),
                "pairwise_top30_intersection_mean": float(
                    np.mean([item["intersection"] for item in pair_overlaps])
                ),
                "pairwise_top30_intersection_minimum": int(
                    min(item["intersection"] for item in pair_overlaps)
                ),
                "pairwise_top30_intersection_maximum": int(
                    max(item["intersection"] for item in pair_overlaps)
                ),
                "all_neutral_consensus_count": len(set.intersection(*sets)),
                "all_neutral_union_count": len(set.union(*sets)),
                "mean_seed_percentile_range": max(seed_means) - min(seed_means),
                "mean_effective_text_percentile_range": max(text_means) - min(text_means),
            }
        )
    return records


def strip_internal_vectors(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: strip_internal_vectors(item)
            for key, item in value.items()
            if key not in {"query_vector", "tangent_vector"}
        }
    if isinstance(value, list):
        return [strip_internal_vectors(item) for item in value]
    return value


def case_record(
    library: composition.ActiveLibrary,
    cache: composition.AnchorCache,
    definition: dict[str, object],
    neutrals: dict[str, np.ndarray],
) -> dict[str, object]:
    seed_id = int(definition["seed_track_id"])
    text_query = str(definition["text"])
    sign = int(definition["sign"])
    seed_reference = f"song:{seed_id}"
    text_reference = f"text:{text_query}"
    seed = cache.embedding(seed_reference)
    text = cache.embedding(text_reference)
    if seed_id not in set(int(value) for value in library.track_ids):
        raise ValueError(f"seed {seed_id} is not active")

    raw = vector_ranking(
        library, cache, seed_reference, text_reference, sign, seed, "raw_seed", 0.0
    )
    all_of_anchors = [
        cache.resolve((seed_reference, float(definition["seed_weight"]))),
        cache.resolve((text_reference, sign * float(definition["text_weight"]))),
    ]
    all_of = composition.evaluate_operator(library, cache, all_of_anchors, "all_of")
    centroid = composition.evaluate_operator(library, cache, all_of_anchors, "direction")
    if all_of["status"] != "ranked" or centroid["status"] != "ranked":
        raise AssertionError("positive seed plus signed text must produce All-of and Direction")
    for baseline in (all_of, centroid):
        baseline["diagnostics"]["duplicates"] = duplicate_diagnostics_for_track_ids(
            library, operator_track_ids(baseline)
        )
    refine = composition.refine_record(
        library,
        cache,
        {
            "id": str(definition["id"]),
            "primary": seed_reference,
            "secondary": text_reference,
            "secondary_sign": sign,
        },
    )
    for width in refine["widths"]:
        width["duplicate_diagnostics"] = duplicate_diagnostics_for_track_ids(
            library, refine_width_track_ids(width)
        )

    signed_target = text if sign > 0 else -text
    slerp_records: list[dict[str, object]] = []
    for angle in ANGLES_DEG:
        query, available_arc = exact_angle_direction(seed, signed_target, angle)
        record = vector_ranking(
            library,
            cache,
            seed_reference,
            text_reference,
            sign,
            query,
            "exact_angle_slerp",
            float(angle),
            {"available_signed_target_arc_degrees": available_arc},
        )
        record["query_vector"] = query.tolist()
        slerp_records.append(record)

    neutral_vectors = dict(neutrals)
    neutral_vectors["normalized_mean"] = unit(
        np.mean(np.stack(list(neutrals.values()), axis=0).astype(np.float64), axis=0)
    )
    residual_records: dict[str, list[dict[str, object]]] = {}
    tangent_vectors: dict[str, np.ndarray] = {}
    for neutral_name, neutral in neutral_vectors.items():
        series: list[dict[str, object]] = []
        for angle in ANGLES_DEG:
            query, tangent = neutral_residual_direction(
                seed, text, neutral, sign, angle
            )
            tangent_vectors[neutral_name] = tangent
            record = vector_ranking(
                library,
                cache,
                seed_reference,
                text_reference,
                sign,
                query,
                "neutral_residual_tangent",
                float(angle),
                {
                    "neutral": neutral_name,
                    "neutral_vector_sha256": sha256_vector(neutral),
                    "tangent_vector_sha256": sha256_vector(tangent),
                },
            )
            record["query_vector"] = query.tolist()
            record["tangent_vector"] = tangent.tolist()
            series.append(record)
        residual_records[neutral_name] = series

    refine_half = next(
        width for width in refine["widths"] if float(width["primary_fraction"]) == 0.005
    )
    baselines = {
        "raw_seed": record_track_ids(raw),
        "production_all_of": operator_track_ids(all_of),
        "old_weighted_centroid": operator_track_ids(centroid),
        "refine_0_5_percent": refine_width_track_ids(refine_half),
    }
    attach_baseline_overlaps(slerp_records, baselines)
    attach_baseline_overlaps(
        itertools.chain.from_iterable(residual_records.values()), baselines
    )

    tangent_cosines = [
        {
            "first": first,
            "second": second,
            "cosine": float(np.dot(tangent_vectors[first], tangent_vectors[second])),
        }
        for first, second in itertools.combinations(sorted(tangent_vectors), 2)
    ]
    result = {
        "id": definition["id"],
        "relation": "with" if sign > 0 else "away",
        "seed": {
            "reference": seed_reference,
            "track_id": seed_id,
            "label": cache.label(seed_reference),
            "embedding_sha256": sha256_vector(seed),
        },
        "text": {
            "reference": text_reference,
            "query": text_query,
            "sign": sign,
            "embedding_sha256": sha256_vector(text),
        },
        "production_weights": {
            "seed": float(definition["seed_weight"]),
            "text_absolute": float(definition["text_weight"]),
        },
        "geometry": {
            "seed_text_cosine": float(np.dot(seed, text)),
            "seed_text_angle_degrees": exact_seed_angle_degrees(seed, text),
            "neutral_tangent_pairwise": tangent_cosines,
        },
        "baselines": {
            "raw_seed": raw,
            "production_all_of": all_of,
            "old_weighted_centroid": centroid,
            "refine": refine,
        },
        "exact_angle_slerp": slerp_records,
        "neutral_residual": residual_records,
        "control_diagnostics": {
            "exact_angle_slerp": series_control_diagnostics(slerp_records),
            "neutral_residual": {
                name: series_control_diagnostics(series)
                for name, series in residual_records.items()
            },
        },
        "neutral_wording_sensitivity": neutral_sensitivity(residual_records),
    }
    return strip_internal_vectors(result)


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else math.nan


def aggregate(cases: Sequence[dict[str, object]]) -> dict[str, object]:
    method_rows: dict[str, list[dict[str, object]]] = {}
    for case in cases:
        method_rows.setdefault("raw_seed", []).append(case["baselines"]["raw_seed"])
        method_rows.setdefault("production_all_of", []).append(
            case["baselines"]["production_all_of"]
        )
        method_rows.setdefault("old_weighted_centroid", []).append(
            case["baselines"]["old_weighted_centroid"]
        )
        for width in case["baselines"]["refine"]["widths"]:
            key = f"refine_{100 * float(width['primary_fraction']):g}_percent"
            method_rows.setdefault(key, []).append(width)
        for record in case["exact_angle_slerp"]:
            key = f"slerp_{float(record['requested_angle_degrees']):g}_degrees"
            method_rows.setdefault(key, []).append(record)
        for neutral_name, series in case["neutral_residual"].items():
            for record in series:
                key = (
                    f"residual_{neutral_name}_"
                    f"{float(record['requested_angle_degrees']):g}_degrees"
                )
                method_rows.setdefault(key, []).append(record)

    summaries: dict[str, object] = {}
    for key, records in sorted(method_rows.items()):
        if key == "production_all_of" or key == "old_weighted_centroid":
            diagnostics = [record["diagnostics"] for record in records]
            summaries[key] = {
                "cases": len(records),
                "mean_seed_percentile": mean(
                    item["mean_effective_percentile_by_anchor"][0] for item in diagnostics
                ),
                "mean_effective_text_percentile": mean(
                    item["mean_effective_percentile_by_anchor"][1] for item in diagnostics
                ),
                "mean_row_worst_percentile": mean(
                    item["mean_row_worst_anchor_percentile"] for item in diagnostics
                ),
                "mean_seed_top100_coverage": mean(
                    item["anchor_top100_coverage"][0] for item in diagnostics
                ),
                "mean_effective_text_top100_coverage": mean(
                    item["anchor_top100_coverage"][1] for item in diagnostics
                ),
                "mean_exact_label_duplicate_excess": mean(
                    item["duplicates"]["exact_artist_title_excess"] for item in diagnostics
                ),
                "mean_embedding_near_duplicate_excess": mean(
                    item["duplicates"]["embedding_ge_0_995_greedy_excess"]
                    for item in diagnostics
                ),
            }
        elif key.startswith("refine_"):
            summaries[key] = {
                "cases": len(records),
                "mean_seed_percentile": mean(
                    record["mean_primary_percentile"] for record in records
                ),
                "mean_effective_text_percentile": mean(
                    record["mean_effective_secondary_percentile"] for record in records
                ),
                "mean_overlap_with_raw_seed_top30": mean(
                    record["overlap_with_raw_primary_top30"] for record in records
                ),
                "mean_exact_label_duplicate_excess": mean(
                    record["duplicate_diagnostics"]["exact_artist_title_excess"]
                    for record in records
                ),
                "mean_embedding_near_duplicate_excess": mean(
                    record["duplicate_diagnostics"]["embedding_ge_0_995_greedy_excess"]
                    for record in records
                ),
            }
        else:
            diagnostics = [record["diagnostics"] for record in records]
            summaries[key] = {
                "cases": len(records),
                "mean_seed_percentile": mean(
                    item["mean_seed_percentile"] for item in diagnostics
                ),
                "mean_effective_text_percentile": mean(
                    item["mean_effective_text_percentile"] for item in diagnostics
                ),
                "mean_row_worst_percentile": mean(
                    item["mean_row_worst_percentile"] for item in diagnostics
                ),
                "mean_seed_top100_coverage": mean(
                    item["seed_top100_coverage"] for item in diagnostics
                ),
                "mean_effective_text_top100_coverage": mean(
                    item["effective_text_top100_coverage"] for item in diagnostics
                ),
                "mean_exact_label_duplicate_excess": mean(
                    item["duplicates"]["exact_artist_title_excess"] for item in diagnostics
                ),
                "mean_embedding_near_duplicate_excess": mean(
                    item["duplicates"]["embedding_ge_0_995_greedy_excess"]
                    for item in diagnostics
                ),
            }
            if all("top30_overlap_with_baselines" in record for record in records):
                summaries[key]["mean_top30_overlap_with_baselines"] = {
                    baseline: mean(
                        record["top30_overlap_with_baselines"][baseline]["intersection"]
                        for record in records
                    )
                    for baseline in (
                        "raw_seed",
                        "production_all_of",
                        "old_weighted_centroid",
                        "refine_0_5_percent",
                    )
                }

    sensitivity_by_angle: list[dict[str, object]] = []
    for angle in ANGLES_DEG:
        rows = [
            next(
                row
                for row in case["neutral_wording_sensitivity"]
                if int(row["angle_degrees"]) == angle
            )
            for case in cases
        ]
        sensitivity_by_angle.append(
            {
                "angle_degrees": angle,
                "mean_pairwise_top30_intersection": mean(
                    row["pairwise_top30_intersection_mean"] for row in rows
                ),
                "worst_pairwise_top30_intersection": min(
                    int(row["pairwise_top30_intersection_minimum"]) for row in rows
                ),
                "mean_all_neutral_consensus_count": mean(
                    row["all_neutral_consensus_count"] for row in rows
                ),
                "mean_all_neutral_union_count": mean(
                    row["all_neutral_union_count"] for row in rows
                ),
                "mean_pairwise_query_cosine": mean(
                    row["pairwise_query_cosine_mean"] for row in rows
                ),
                "worst_pairwise_query_cosine": min(
                    float(row["pairwise_query_cosine_minimum"]) for row in rows
                ),
            }
        )

    monotonic: dict[str, object] = {}
    families = {
        "exact_angle_slerp": [
            case["control_diagnostics"]["exact_angle_slerp"] for case in cases
        ]
    }
    neutral_names = sorted(cases[0]["control_diagnostics"]["neutral_residual"])
    for neutral_name in neutral_names:
        families[f"neutral_residual/{neutral_name}"] = [
            case["control_diagnostics"]["neutral_residual"][neutral_name]
            for case in cases
        ]
    for family, records in families.items():
        monotonic[family] = {
            "case_count": len(records),
            "query_effective_text_violations": sum(
                int(record["query_effective_text_monotonic_increase_violations"])
                for record in records
            ),
            "selected_seed_violations": sum(
                int(record["selected_seed_monotonic_decrease_violations"])
                for record in records
            ),
            "selected_effective_text_violations": sum(
                int(record["selected_effective_text_monotonic_increase_violations"])
                for record in records
            ),
            "possible_adjacent_steps": len(records) * (len(ANGLES_DEG) - 1),
        }
    return {
        "method_means": summaries,
        "neutral_wording_sensitivity_by_angle": sensitivity_by_angle,
        "monotonicity": monotonic,
    }


def track_label(row: dict[str, object]) -> str:
    return f"{row['artist'] or '?'} - {row['title'] or '?'}".replace("|", "/")


def list_lines(label: str, record: dict[str, object], count: int = 10) -> list[str]:
    lines = [f"### {label}", "", "| # | Track | Seed pct | Text pct | Score |", "|---:|---|---:|---:|---:|"]
    for row in record["top30"][:count]:
        lines.append(
            f"| {row['rank']} | {track_label(row)} | "
            f"{float(row['effective_anchor_percentiles'][0]):.5f} | "
            f"{float(row['effective_anchor_percentiles'][1]):.5f} | "
            f"{float(row['objective_score']):.6f} |"
        )
    lines.append("")
    return lines


def write_qualitative(path: Path, cases: Sequence[dict[str, object]]) -> None:
    lines = [
        "# Residual Direction Qualitative Lists",
        "",
        "Every ranking uses embeddings only. Labels are diagnostic. Text pct is inverted for Away.",
        "",
    ]
    for case in cases:
        lines += [
            f"## {case['id']}",
            "",
            f"Seed: {case['seed']['label']}",
            f"Text: {case['relation']} `{case['text']['query']}`",
            f"Seed/text cosine: {float(case['geometry']['seed_text_cosine']):.6f}",
            "",
        ]
        lines += list_lines("Raw seed", case["baselines"]["raw_seed"])
        lines += list_lines("Production All-of", case["baselines"]["production_all_of"])
        lines += list_lines("Old weighted centroid", case["baselines"]["old_weighted_centroid"])
        for width in case["baselines"]["refine"]["widths"]:
            lines += list_lines(
                f"Refine {100 * float(width['primary_fraction']):g}%", width
            )
        for record in case["exact_angle_slerp"]:
            lines += list_lines(
                f"Exact-angle slerp {float(record['requested_angle_degrees']):g} deg",
                record,
            )
        for neutral_name, records in case["neutral_residual"].items():
            for record in records:
                lines += list_lines(
                    f"Residual {neutral_name} {float(record['requested_angle_degrees']):g} deg",
                    record,
                )
        lines += ["### Neutral wording sensitivity", ""]
        lines += [
            "| Angle | Mean/min pair overlap | Consensus / union | Mean/min query cosine |",
            "|---:|---:|---:|---:|",
        ]
        for row in case["neutral_wording_sensitivity"]:
            lines.append(
                f"| {row['angle_degrees']} | "
                f"{float(row['pairwise_top30_intersection_mean']):.2f} / "
                f"{row['pairwise_top30_intersection_minimum']} | "
                f"{row['all_neutral_consensus_count']} / {row['all_neutral_union_count']} | "
                f"{float(row['pairwise_query_cosine_mean']):.5f} / "
                f"{float(row['pairwise_query_cosine_minimum']):.5f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=composition.DEFAULT_DB)
    parser.add_argument("--active-catalog", type=Path, default=composition.DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--phone-report", type=Path, default=composition.DEFAULT_PHONE_REPORT)
    parser.add_argument("--text-results", type=Path, default=DEFAULT_TEXT_RESULTS)
    parser.add_argument("--neutral-results", type=Path, default=DEFAULT_NEUTRAL_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    catalog = composition.parse_active_catalog(args.active_catalog)
    library, db_hash = composition.load_active_library(
        args.db, catalog, verify_hash=not args.skip_db_hash
    )
    load_seconds = time.perf_counter() - total_started

    phone_embeddings, phone_hashes, _ = composition.load_phone_text_embeddings(
        args.phone_report
    )
    legacy_embeddings, legacy_provenance = load_prompt_jsonl(args.text_results)
    neutral_embeddings, neutral_provenance = load_prompt_jsonl(args.neutral_results)
    required_text = {str(case["text"]) for case in CASES}
    text_embeddings: dict[str, np.ndarray] = {}
    text_provenance: dict[str, dict[str, object]] = {}
    for query in sorted(required_text):
        if query in phone_embeddings:
            text_embeddings[query] = unit(phone_embeddings[query])
            text_provenance[query] = {
                "source": str(args.phone_report),
                "source_sha256": composition.sha256_file(args.phone_report),
                "kind": "exact repeated phone embedding",
                "embedding_sha256": phone_hashes[query],
            }
        elif query in legacy_embeddings:
            text_embeddings[query] = legacy_embeddings[query]
            text_provenance[query] = legacy_provenance[query]
        else:
            raise ValueError(f"no frozen text embedding for {query!r}")

    neutral_by_name = {
        str(provenance["row_id"]).split("/", 1)[-1]: neutral_embeddings[prompt]
        for prompt, provenance in neutral_provenance.items()
    }
    if len(neutral_by_name) != 4:
        raise ValueError("exactly four frozen neutral prompts are required")
    cache = composition.AnchorCache(library, text_embeddings)

    evaluation_started = time.perf_counter()
    cases = []
    for position, definition in enumerate(CASES, start=1):
        record = case_record(library, cache, definition, neutral_by_name)
        cases.append(record)
        print(f"case {position}/{len(CASES)} {definition['id']}", flush=True)
    summary = aggregate(cases)
    evaluation_seconds = time.perf_counter() - evaluation_started

    definitions = {
        "cases": CASES,
        "angles_degrees": ANGLES_DEG,
        "refine_widths": REFINE_WIDTHS,
        "top_k": TOP_K,
        "diagnostic_k": DIAGNOSTIC_K,
        "near_duplicate_cosine": NEAR_DUPLICATE_COSINE,
    }
    results: dict[str, object] = {
        "schema": "residual-direction-eval-v1",
        "scope": {
            "ranking_input": "CLaMP3 embeddings only",
            "metadata": "qualitative labels and duplicate diagnostics only",
            "claim_boundary": "PDV-like directional prompting is inspiration, not validation for CLaMP3",
        },
        "inputs": {
            "database": str(args.db.resolve()),
            "database_sha256": db_hash,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": composition.sha256_file(args.active_catalog),
            "active_tracks": library.count,
            "text_provenance": text_provenance,
            "neutral_provenance": neutral_provenance,
        },
        "contracts": {
            "slerp": "exact signed angular move from seed on the seed/text great circle",
            "residual": "d=text-neutral; tangent=d-(d dot seed)seed; exact signed angular move",
            "all_of": "production weighted geometric mean of signed empirical percentiles",
            "refine": "effective text order inside an explicit seed percentile neighborhood",
            "tie_break": "ascending active track ID after Float32 objective score",
            "seed_exclusion": "the seed track ID is excluded from every result list",
        },
        "definitions_sha256": composition.sha256_json(definitions),
        "definitions": definitions,
        "summary": summary,
        "cases": cases,
    }
    results["deterministic_payload_sha256"] = composition.sha256_json(results)

    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "results.json"
    qualitative_path = args.output / "qualitative-lists.md"
    composition.atomic_json(results_path, results)
    write_qualitative(qualitative_path, cases)
    manifest = {
        "schema": "residual-direction-run-manifest-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runtime_seconds": {
            "load_and_active_projection": load_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_started,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "deterministic_payload_sha256": results["deterministic_payload_sha256"],
        "artifacts": {
            "results.json": composition.sha256_file(results_path),
            "qualitative-lists.md": composition.sha256_file(qualitative_path),
        },
    }
    composition.atomic_json(args.output / "run-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
