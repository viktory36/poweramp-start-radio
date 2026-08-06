#!/usr/bin/env python3
"""Deep text/composed-retrieval audit using explicit live-phone usage evidence.

This is experiment code, not an app implementation. Audio/text embeddings alone define
relevance. Metadata is used only to make results readable and to audit identity crowding;
it never enters a ranking score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import v2_queue_eval as queue_eval
from poweramp_indexer.embeddings_clamp3 import (
    CLAMP3_WEIGHTS_FILENAME,
    CLaMP3EmbeddingGenerator,
)


REPO_ROOT = queue_eval.REPO_ROOT
DEFAULT_PHONE_SNAPSHOT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-live"
    / "2026-07-13T234337+0300"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text-retrieval-deep"
)

# Declared before this experiment's rankings were generated. The first group comes from
# the live phone's explicit recent/session queries; the second group is diagnostic.
DECLARED_QUERIES: tuple[tuple[str, str], ...] = (
    ("sleep", "user_stated"),
    ("psytrance", "negation_control"),
    ("electronic sitar", "composition_control"),
    ("sitar", "composition_atom"),
    ("electronic", "composition_atom"),
    ("dark techno", "composition_control"),
    ("energetic", "composition_control"),
    ("trance", "composition_control"),
    ("music that is not psytrance", "negation_control"),
    ("music without psytrance", "negation_control"),
    ("easy listening at 2am", "paraphrase_control"),
)

DEPTHS: tuple[int, ...] = (10, 20, 30, 50, 100, 200, 500)
RAW_THRESHOLDS: tuple[float, ...] = (0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30)
WEIGHT_SWEEP: tuple[float, ...] = (
    0.0,
    0.01,
    0.02,
    0.05,
    0.10,
    0.18,
    0.25,
    1.0 / 3.0,
    0.50,
    2.0 / 3.0,
    0.82,
    0.90,
    0.95,
    0.98,
    0.99,
    1.0,
)
REFINE_FRACTIONS: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)

ACTUAL_COMPOSITION_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "bonobo_sleep",
        "anchors": (("song", 80437, 0.82), ("text", "sleep", 0.18)),
        "primary": 0,
        "secondary": 1,
        "source": "user_stated_and_session_45",
    },
    {
        "id": "bonobo_time",
        "anchors": (("song", 80437, 2.0 / 3.0), ("song", 75611, 1.0 / 3.0)),
        "primary": 0,
        "secondary": 1,
        "source": "recent_search_2",
    },
    {
        "id": "downtempo_jamming",
        "anchors": (("text", "downtempo jazz", 0.5), ("song", 5019, 0.5)),
        "primary": 0,
        "secondary": 1,
        "source": "recent_search_6_and_session_51",
    },
    {
        "id": "sitar_avoid_psy",
        "anchors": (("text", "sitar, electronic", 0.5), ("song", 1319, -0.5)),
        "primary": 0,
        "secondary": 1,
        "source": "recent_search_8",
    },
    {
        "id": "bonobo_time_shpongle",
        "anchors": (
            ("song", 80437, 0.5),
            ("song", 75611, 0.25),
            ("song", 26980, 0.25),
        ),
        "source": "recent_search_1_and_session_54",
    },
    {
        "id": "soulful_three_song",
        "anchors": (
            ("song", 25528, 0.5),
            ("song", 1670, 0.25),
            ("song", 41266, 0.25),
        ),
        "source": "recent_search_3_and_session_52",
    },
    {
        "id": "sitar_electronic_text_atoms",
        "anchors": (("text", "sitar", 0.5), ("text", "electronic", 0.5)),
        "whole_query": "sitar, electronic",
        "source": "recent_search_9_phrase_vs_explicit_atoms",
    },
    {
        "id": "dark_techno_energetic_trance_text_atoms",
        "anchors": (
            ("text", "dark techno", 1.0 / 3.0),
            ("text", "energetic", 1.0 / 3.0),
            ("text", "trance", 1.0 / 3.0),
        ),
        "whole_query": "dark techno, energetic, trance",
        "source": "phone_direct_queue_phrase_vs_explicit_atoms",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def load_settings(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    result: dict[str, object] = {}
    converters = {
        "int": int,
        "long": int,
        "float": float,
        "boolean": lambda value: value.lower() == "true",
    }
    for child in root:
        name = child.attrib.get("name")
        if not name:
            continue
        if child.tag == "string":
            result[name] = child.text or ""
        elif child.tag in converters:
            result[name] = converters[child.tag](child.attrib["value"])
    return result


def result_label(recent: dict[str, object]) -> str:
    """Mirror the result label, not the shorter recent-search display label."""
    text = str(recent.get("text", ""))
    seeds = list(recent.get("seeds", []))
    if not seeds:
        return text

    parts: list[tuple[str, bool]] = []
    if text.strip():
        weight = round(float(recent.get("text_weight", 1.0)) * 100)
        parts.append((f"{text} ({weight}%)", bool(recent.get("text_negative", False))))
    for seed in seeds:
        weight = round(float(seed["weight"]) * 100)
        parts.append(
            (
                f"{seed.get('title', '?')} ({weight}%)",
                bool(seed.get("negative", False)),
            )
        )
    if not parts:
        return ""
    rendered = ("- " if parts[0][1] else "") + parts[0][0]
    for text_part, negative in parts[1:]:
        rendered += (" - " if negative else " + ") + text_part
    return rendered


def parse_phone_usage(snapshot: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    settings_path = snapshot / "shared_prefs" / "settings.xml"
    sessions_path = snapshot / "files" / "session_history.json"
    settings = load_settings(settings_path)
    recent = json.loads(str(settings.get("recent_searches_v2", "[]")))
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    direct = [row for row in sessions if row.get("isDirectQueue", False)]
    recent_labels = {result_label(row) for row in recent}

    identity_excess = Counter()
    sessions_with_identity_excess = Counter()
    unique_id_deficit = 0
    sessions_with_unique_id_deficit = 0
    for session in direct:
        tracks = [row["track"] for row in session["tracks"]]
        for field in ("metadataKey", "filenameKey", "filePath"):
            values = [str(track.get(field, "")).casefold() for track in tracks]
            excess = len(values) - len(set(values))
            identity_excess[field] += excess
            sessions_with_identity_excess[field] += excess > 0
        deficit = len(tracks) - len(session.get("queuedFileIds", []))
        unique_id_deficit += deficit
        sessions_with_unique_id_deficit += deficit > 0

    direct_labels = [str(row["seedTrack"].get("title", "")) for row in direct]
    summary = {
        "snapshot": str(snapshot.resolve()),
        "settings_sha256": sha256_file(settings_path),
        "session_history_sha256": sha256_file(sessions_path),
        "session_count": len(sessions),
        "direct_result_queue_count": len(direct),
        "direct_result_queue_fraction": len(direct) / len(sessions) if sessions else 0.0,
        "radio_mode_counts": dict(
            Counter(
                row.get("config", {}).get("selectionMode", "UNKNOWN")
                for row in sessions
                if not row.get("isDirectQueue", False)
            )
        ),
        "direct_pure_text_count": sum("%" not in label for label in direct_labels),
        "direct_composed_count": sum("%" in label for label in direct_labels),
        "direct_track_rows": sum(len(row["tracks"]) for row in direct),
        "direct_unique_poweramp_id_sets": sum(
            len(row.get("queuedFileIds", [])) for row in direct
        ),
        "direct_unique_id_deficit": unique_id_deficit,
        "direct_sessions_with_unique_id_deficit": sessions_with_unique_id_deficit,
        "identity_duplicate_excess": dict(identity_excess),
        "sessions_with_identity_duplicate_excess": dict(sessions_with_identity_excess),
        "recent_search_count": len(recent),
        "direct_sessions_recoverable_from_current_recent_label": sum(
            label in recent_labels for label in direct_labels
        ),
        "current_text_search_top_k": settings.get("text_search_top_k"),
        "recent_searches": recent,
        "direct_labels": direct_labels,
    }
    return summary, sessions


def collect_prompt_rows(
    usage: dict[str, object], sessions: Sequence[dict[str, object]]
) -> list[dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}

    def add(query: str, source: str) -> None:
        normalized = query.strip()
        if not normalized:
            return
        key = normalized.casefold()
        rows.setdefault(key, {"query": normalized, "source": source})

    for recent in usage["recent_searches"]:
        add(str(recent.get("text", "")), "phone_recent_search")
    for session in sessions:
        if not session.get("isDirectQueue", False):
            continue
        label = str(session["seedTrack"].get("title", ""))
        if "%" not in label:
            add(label, "phone_direct_queue")
    for query, source in DECLARED_QUERIES:
        add(query, source)
    return sorted(rows.values(), key=lambda row: row["query"].casefold())


def canonical_order(scores: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    return np.lexsort((track_ids, -scores))


def percentile_goodness(
    scores: np.ndarray, track_ids: np.ndarray, positive: bool = True
) -> np.ndarray:
    """Return (0, 1] percentiles where 1 is best for the requested direction."""
    objective = scores if positive else -scores
    ascending = np.lexsort((track_ids, objective))
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[ascending] = np.arange(1, scores.size + 1)
    return ranks.astype(np.float64) / scores.size


def power_mean_objective(
    percentiles: Sequence[np.ndarray], weights: Sequence[float], power: float
) -> np.ndarray:
    if not percentiles or len(percentiles) != len(weights):
        raise ValueError("percentiles and weights must be nonempty and equal length")
    magnitudes = np.abs(np.asarray(weights, dtype=np.float64))
    if not np.isfinite(magnitudes).all() or magnitudes.sum() <= 0:
        raise ValueError("weights must contain a finite nonzero magnitude")
    active = magnitudes > 0
    magnitudes = magnitudes[active]
    magnitudes /= magnitudes.sum()
    matrix = np.stack(percentiles).astype(np.float64, copy=False)[active]
    if np.any(matrix <= 0) or np.any(matrix > 1):
        raise ValueError("percentiles must lie in (0, 1]")
    if math.isinf(power) and power < 0:
        return matrix.min(axis=0)
    if abs(power) < 1e-12:
        return np.exp(np.sum(magnitudes[:, None] * np.log(matrix), axis=0))
    return np.power(
        np.sum(magnitudes[:, None] * np.power(matrix, power), axis=0),
        1.0 / power,
    )


def weighted_union_indices(
    percentiles: Sequence[np.ndarray],
    weights: Sequence[float],
    track_ids: np.ndarray,
    count: int,
    excluded_track_ids: Iterable[int] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Interleave positive anchor branches with deterministic weighted fairness.

    Unlike a max score, this preserves the stated branch allocation even when anchor
    score distributions or weights differ. A track present in several branches appears
    once, and the branch whose turn selected it owns that row.
    """
    if not percentiles or len(percentiles) != len(weights):
        raise ValueError("percentiles and weights must be nonempty and equal length")
    magnitudes = np.asarray(weights, dtype=np.float64)
    if np.any(magnitudes < 0):
        raise ValueError("weighted union does not accept negative anchors")
    active = np.flatnonzero(magnitudes > 0)
    if active.size == 0:
        raise ValueError("weighted union needs at least one positive anchor")
    shares = magnitudes[active] / magnitudes[active].sum()
    orders = [canonical_order(percentiles[int(index)], track_ids) for index in active]
    excluded = set(int(value) for value in excluded_track_ids)
    cursors = np.zeros(active.size, dtype=np.int64)
    allocations = np.zeros(active.size, dtype=np.int64)
    selected: list[int] = []
    origins: list[int] = []
    selected_track_ids: set[int] = set()

    while len(selected) < count:
        # Largest current quota deficit gives proportional prefixes, not just a final
        # proportional count. Lower anchor position is the stable tie-break.
        target = shares * (len(selected) + 1)
        branch_order = np.lexsort((np.arange(active.size), -(target - allocations)))
        picked = False
        for raw_branch in branch_order:
            branch = int(raw_branch)
            order = orders[branch]
            while cursors[branch] < order.size:
                index = int(order[cursors[branch]])
                cursors[branch] += 1
                track_id = int(track_ids[index])
                if track_id in excluded or track_id in selected_track_ids:
                    continue
                selected.append(index)
                origins.append(int(active[branch]))
                selected_track_ids.add(track_id)
                allocations[branch] += 1
                picked = True
                break
            if picked:
                break
        if not picked:
            break
    return np.asarray(selected, dtype=np.int64), np.asarray(origins, dtype=np.int64)


def top_indices(
    objective: np.ndarray,
    library: queue_eval.Library,
    count: int,
    excluded_track_ids: Iterable[int] = (),
) -> np.ndarray:
    scores = objective.copy()
    excluded = set(int(value) for value in excluded_track_ids)
    if excluded:
        mask = np.fromiter(
            (int(track_id) in excluded for track_id in library.track_ids),
            dtype=bool,
            count=library.count,
        )
        scores[mask] = -np.inf
    return canonical_order(scores, library.track_ids)[:count]


def set_overlap(left: Sequence[int], right: Sequence[int]) -> dict[str, float | int]:
    a = set(int(value) for value in left)
    b = set(int(value) for value in right)
    intersection = len(a & b)
    union = len(a | b)
    return {
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
    }


def identity_metrics(library: queue_eval.Library, indices: Sequence[int]) -> dict[str, int]:
    result: dict[str, int] = {"count": len(indices)}
    for name, values in (
        ("metadata_key", library.metadata_keys),
        ("filename_key", library.filename_keys),
        ("file_path", library.file_paths),
    ):
        selected = [values[int(index)].casefold() for index in indices]
        result[f"{name}_duplicate_excess"] = len(selected) - len(set(selected))
    return result


def unique_fill(
    library: queue_eval.Library,
    ordered_indices: Sequence[int],
    count: int,
    identity: str,
) -> tuple[list[int], int]:
    values = {
        "metadata_key": library.metadata_keys,
        "filename_key": library.filename_keys,
        "file_path": library.file_paths,
    }[identity]
    selected: list[int] = []
    seen: set[str] = set()
    scanned = 0
    for raw_index in ordered_indices:
        scanned += 1
        index = int(raw_index)
        key = values[index].casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(index)
        if len(selected) == count:
            break
    return selected, scanned


def result_rows(
    library: queue_eval.Library,
    indices: Sequence[int],
    objective: np.ndarray,
    per_anchor: Sequence[np.ndarray] = (),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, raw_index in enumerate(indices, start=1):
        index = int(raw_index)
        rows.append(
            {
                "rank": rank,
                "track_id": int(library.track_ids[index]),
                "objective": float(objective[index]),
                "anchor_percentiles": [float(values[index]) for values in per_anchor],
                "artist": library.artists[index],
                "album": library.albums[index],
                "title": library.titles[index],
                "duration_ms": int(library.durations_ms[index]),
                "metadata_key": library.metadata_keys[index],
                "filename_key": library.filename_keys[index],
                "file_path": library.file_paths[index],
            }
        )
    return rows


def union_result_rows(
    library: queue_eval.Library,
    indices: Sequence[int],
    origins: Sequence[int],
    per_anchor: Sequence[np.ndarray],
) -> list[dict[str, object]]:
    rows = result_rows(
        library,
        indices,
        np.zeros(library.count, dtype=np.float32),
        per_anchor,
    )
    for row, origin in zip(rows, origins, strict=True):
        row.pop("objective")
        row["branch_origin"] = int(origin)
    return rows


def largest_gap_rank(sorted_scores: np.ndarray, minimum: int = 5, maximum: int = 500) -> dict[str, float | int]:
    if sorted_scores.size < minimum + 1:
        raise ValueError("not enough scores")
    stop = min(maximum, sorted_scores.size - 1)
    gaps = sorted_scores[minimum - 1 : stop] - sorted_scores[minimum: stop + 1]
    offset = int(np.argmax(gaps))
    rank = minimum + offset
    return {"rank": rank, "gap": float(gaps[offset])}


def score_depth_record(
    library: queue_eval.Library, query: str, embedding: np.ndarray
) -> dict[str, object]:
    similarities = library.embeddings @ embedding
    order = canonical_order(similarities, library.track_ids)
    sorted_scores = similarities[order]
    depths: dict[str, object] = {}
    for depth in DEPTHS:
        prefix = order[:depth]
        _, metadata_scan = unique_fill(
            library, order, depth, "metadata_key"
        )
        _, filename_scan = unique_fill(
            library, order, depth, "filename_key"
        )
        depths[str(depth)] = {
            "boundary_score": float(sorted_scores[depth - 1]),
            "next_score": float(sorted_scores[depth]),
            "boundary_gap": float(sorted_scores[depth - 1] - sorted_scores[depth]),
            "identity": identity_metrics(library, prefix),
            "metadata_unique_scan_depth": metadata_scan,
            "filename_unique_scan_depth": filename_scan,
            "metadata_unique_replacements": metadata_scan - depth,
            "filename_unique_replacements": filename_scan - depth,
            "mean_pairwise_to_query": float(np.mean(similarities[prefix])),
        }
    percentiles = np.percentile(similarities, [0, 1, 10, 50, 90, 99, 99.9, 100])
    return {
        "query": query,
        "score_distribution": {
            key: float(value)
            for key, value in zip(
                ("min", "p01", "p10", "p50", "p90", "p99", "p999", "max"),
                percentiles,
                strict=True,
            )
        },
        "raw_threshold_counts": {
            str(threshold): int(np.count_nonzero(similarities >= threshold))
            for threshold in RAW_THRESHOLDS
        },
        "largest_gap_rank_5_500": largest_gap_rank(sorted_scores),
        "depths": depths,
        "top_100": result_rows(library, order[:100], similarities),
    }


def append_embedding_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_embedding_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query = str(row["query"])
            embedding = np.asarray(row["embedding"], dtype=np.float32)
            if embedding.shape != (queue_eval.EXPECTED_DIM,):
                raise ValueError(f"checkpoint line {line_number} has shape {embedding.shape}")
            result[query.casefold()] = row
    return result


def embed_prompts(
    rows: Sequence[dict[str, str]], checkpoint: Path, force: bool
) -> dict[str, np.ndarray]:
    if force and checkpoint.exists():
        checkpoint.unlink()
    completed = load_embedding_checkpoint(checkpoint)
    missing = [row for row in rows if row["query"].casefold() not in completed]
    if missing:
        generator = CLaMP3EmbeddingGenerator(fp16=False)
        try:
            for position, row in enumerate(missing, start=1):
                started = time.perf_counter()
                values = generator.embed_text(row["query"])
                if values is None:
                    raise RuntimeError(f"text embedding failed: {row['query']}")
                embedding = np.asarray(values, dtype=np.float32)
                record = {
                    **row,
                    "embedding": embedding.tolist(),
                    "embedding_norm": float(np.linalg.norm(embedding)),
                    "inference_ms": (time.perf_counter() - started) * 1000.0,
                }
                append_embedding_checkpoint(checkpoint, record)
                completed[row["query"].casefold()] = record
                print(
                    f"text {position}/{len(missing)}: {row['query']}", file=sys.stderr
                )
        finally:
            generator.unload_models()
    return {
        row["query"].casefold(): np.asarray(
            completed[row["query"].casefold()]["embedding"], dtype=np.float32
        )
        for row in rows
    }


def resolve_anchor(
    anchor: tuple[object, object, object],
    library: queue_eval.Library,
    text_embeddings: dict[str, np.ndarray],
    index_by_track_id: dict[int, int],
) -> tuple[str, np.ndarray, float, int | None]:
    kind, value, weight = anchor
    if kind == "text":
        query = str(value)
        return query, text_embeddings[query.casefold()], float(weight), None
    track_id = int(value)
    index = index_by_track_id[track_id]
    label = f"{library.artists[index] or '?'} - {library.titles[index] or '?'}"
    return label, library.embeddings[index], float(weight), track_id


def composition_record(
    case: dict[str, object],
    library: queue_eval.Library,
    text_embeddings: dict[str, np.ndarray],
    index_by_track_id: dict[int, int],
) -> dict[str, object]:
    anchors = [
        resolve_anchor(raw, library, text_embeddings, index_by_track_id)
        for raw in case["anchors"]
    ]
    similarities = [library.embeddings @ anchor[1] for anchor in anchors]
    percentiles = [
        percentile_goodness(scores, library.track_ids, positive=anchor[2] >= 0)
        for scores, anchor in zip(similarities, anchors, strict=True)
    ]
    weights = [anchor[2] for anchor in anchors]
    excluded = [anchor[3] for anchor in anchors if anchor[3] is not None]
    methods = {
        "strict_min": float("-inf"),
        "strict_p_minus_4": -4.0,
        "harmonic": -1.0,
        "geo_v1": 0.0,
        "arithmetic": 1.0,
    }
    method_rows: dict[str, object] = {}
    top_by_method: dict[str, np.ndarray] = {}
    for name, power in methods.items():
        objective = power_mean_objective(percentiles, weights, power)
        top = top_indices(objective, library, 30, excluded)
        top_by_method[name] = top
        anchor_means = [float(np.mean(values[top])) for values in percentiles]
        method_rows[name] = {
            "anchor_mean_percentiles": anchor_means,
            "mean_worst_anchor_percentile": float(
                np.mean(np.min(np.stack(percentiles)[:, top], axis=0))
            ),
            "identity": identity_metrics(library, top),
            "top_30": result_rows(library, top, objective, percentiles),
        }


    if all(weight >= 0 for weight in weights) and any(weight > 0 for weight in weights):
        union_top, union_origins = weighted_union_indices(
            percentiles,
            weights,
            library.track_ids,
            30,
            excluded,
        )
        top_by_method["either_weighted_union"] = union_top
        method_rows["either_weighted_union"] = {
            "anchor_mean_percentiles": [
                float(np.mean(values[union_top])) for values in percentiles
            ],
            "mean_worst_anchor_percentile": float(
                np.mean(np.min(np.stack(percentiles)[:, union_top], axis=0))
            ),
            "branch_counts": {
                str(anchor_index): int(np.count_nonzero(union_origins == anchor_index))
                for anchor_index in range(len(anchors))
            },
            "identity": identity_metrics(library, union_top),
            "top_30": union_result_rows(
                library, union_top, union_origins, percentiles
            ),
        }
    overlaps: dict[str, object] = {}
    names = list(top_by_method)
    for left_position, left in enumerate(names):
        for right in names[left_position + 1 :]:
            overlaps[f"{left}__{right}"] = set_overlap(
                top_by_method[left], top_by_method[right]
            )

    record: dict[str, object] = {
        "id": case["id"],
        "source": case["source"],
        "anchors": [
            {
                "label": anchor[0],
                "weight": anchor[2],
                "track_id": anchor[3],
            }
            for anchor in anchors
        ],
        "methods": method_rows,
        "method_overlaps": overlaps,
    }


    whole_query = case.get("whole_query")
    if whole_query is not None:
        query = str(whole_query)
        whole_scores = library.embeddings @ text_embeddings[query.casefold()]
        whole_top = top_indices(whole_scores, library, 30, excluded)
        comparison: dict[str, object] = {
            "query": query,
            "geo_top_30_overlap": set_overlap(whole_top, top_by_method["geo_v1"]),
            "whole_phrase_top_30": result_rows(
                library, whole_top, whole_scores, percentiles
            ),
        }
        if "either_weighted_union" in top_by_method:
            comparison["either_top_30_overlap"] = set_overlap(
                whole_top, top_by_method["either_weighted_union"]
            )
        record["whole_query_comparison"] = comparison

    if len(anchors) == 2:
        sweep: list[dict[str, object]] = []
        previous_top: np.ndarray | None = None
        for second_weight in WEIGHT_SWEEP:
            objective = power_mean_objective(
                percentiles, (1.0 - second_weight, second_weight), 0.0
            )
            top = top_indices(objective, library, 30, excluded)
            sweep.append(
                {
                    "second_weight": second_weight,
                    "anchor_mean_percentiles": [
                        float(np.mean(values[top])) for values in percentiles
                    ],
                    "previous_top_30_overlap": (
                        None if previous_top is None else set_overlap(previous_top, top)
                    ),
                    "top_track_ids": [int(library.track_ids[index]) for index in top],
                }
            )
            previous_top = top
        record["geo_weight_sweep"] = sweep

        primary_index = int(case.get("primary", 0))
        secondary_index = int(case.get("secondary", 1))
        primary_order = canonical_order(
            percentiles[primary_index], library.track_ids
        )
        refine: list[dict[str, object]] = []
        for fraction in REFINE_FRACTIONS:
            pool_count = max(1, math.ceil(library.count * fraction))
            pool = primary_order[:pool_count]
            pool = np.asarray(
                [index for index in pool if int(library.track_ids[index]) not in excluded],
                dtype=np.int64,
            )
            secondary = percentiles[secondary_index]
            local_order = np.lexsort(
                (library.track_ids[pool], -secondary[pool])
            )
            top = pool[local_order[:30]]
            refine.append(
                {
                    "fraction": fraction,
                    "pool_count": pool_count,
                    "anchor_mean_percentiles": [
                        float(np.mean(values[top])) for values in percentiles
                    ],
                    "primary_min_percentile": float(
                        np.min(percentiles[primary_index][top])
                    ),
                    "identity": identity_metrics(library, top),
                    "top_30": result_rows(
                        library, top, secondary, percentiles
                    ),
                }
            )
        record["refine"] = refine
    return record


def query_pair_record(
    left: str,
    right: str,
    embeddings: dict[str, np.ndarray],
    library: queue_eval.Library,
) -> dict[str, object]:
    a = embeddings[left.casefold()]
    b = embeddings[right.casefold()]
    a_scores = library.embeddings @ a
    b_scores = library.embeddings @ b
    a_order = canonical_order(a_scores, library.track_ids)
    b_order = canonical_order(b_scores, library.track_ids)
    return {
        "left": left,
        "right": right,
        "embedding_cosine": float(a @ b),
        "top_30": set_overlap(a_order[:30], b_order[:30]),
        "top_100": set_overlap(a_order[:100], b_order[:100]),
    }


def qualitative_markdown(
    usage: dict[str, object],
    depth: Sequence[dict[str, object]],
    compositions: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# Deep Text Retrieval Qualitative Packet",
        "",
        "Metadata below is for human inspection only. It did not enter ranking.",
        "",
        "## Live Usage",
        "",
        f"- Sessions: {usage['session_count']}",
        f"- Direct result-list queues: {usage['direct_result_queue_count']}",
        f"- Direct composed queues: {usage['direct_composed_count']}",
        f"- Direct pure-text queues: {usage['direct_pure_text_count']}",
        "",
    ]
    for row in depth:
        lines.extend((f"## Text: {row['query']}", ""))
        for result in row["top_100"][:30]:
            lines.append(
                f"{result['rank']:>2}. {result['artist'] or '?'} - "
                f"{result['title'] or '?'} ({result['objective']:.4f})"
            )
        lines.append("")
    for case in compositions:
        lines.extend((f"## Composition: {case['id']}", ""))
        methods = ["geo_v1", "strict_min", "harmonic", "arithmetic"]
        if "either_weighted_union" in case["methods"]:
            methods.append("either_weighted_union")
        for method in methods:
            lines.extend((f"### {method}", ""))
            for result in case["methods"][method]["top_30"][:15]:
                anchor_text = ", ".join(
                    f"{value:.4f}" for value in result["anchor_percentiles"]
                )
                suffix = ""
                if "branch_origin" in result:
                    suffix = f" branch={result['branch_origin']}"
                lines.append(
                    f"{result['rank']:>2}. {result['artist'] or '?'} - "
                    f"{result['title'] or '?'} [anchors {anchor_text}{suffix}]"
                )
            lines.append("")
        comparison = case.get("whole_query_comparison")
        if comparison is not None:
            lines.extend((f"### whole phrase: {comparison['query']}", ""))
            for result in comparison["whole_phrase_top_30"][:15]:
                lines.append(
                    f"{result['rank']:>2}. {result['artist'] or '?'} - "
                    f"{result['title'] or '?'}"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def resolve_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    ).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--phone-snapshot", type=Path, default=DEFAULT_PHONE_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    usage, sessions = parse_phone_usage(args.phone_snapshot)
    prompts = collect_prompt_rows(usage, sessions)
    library, db_hash = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "text-embeddings.jsonl"
    text_embeddings = embed_prompts(prompts, checkpoint, force=args.force)

    depth = [
        score_depth_record(library, row["query"], text_embeddings[row["query"].casefold()])
        for row in prompts
    ]
    index_by_track_id = {
        int(track_id): index for index, track_id in enumerate(library.track_ids)
    }
    compositions = [
        composition_record(case, library, text_embeddings, index_by_track_id)
        for case in ACTUAL_COMPOSITION_CASES
    ]

    pair_specs = (
        ("easy listening 2am", "easy listening at night"),
        ("easy listening at night", "easy listening for night"),
        ("easy listening", "soothing easy listening"),
        ("left field", "absurd leftfield"),
        ("sitar, electronic", "electronic sitar"),
        ("psytrance", "not psytrance"),
        ("psytrance", "music that is not psytrance"),
        ("psytrance", "music without psytrance"),
    )
    query_pairs = [
        query_pair_record(left, right, text_embeddings, library)
        for left, right in pair_specs
    ]

    model_path = resolve_checkpoint()
    manifest = {
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "track_count": library.count,
            "dim": library.dim,
        },
        "phone_usage": usage,
        "model": {
            "name": "sander-wood/clamp3",
            "checkpoint_sha256": sha256_file(model_path),
        },
        "prompt_count": len(prompts),
        "prompt_sources": dict(Counter(row["source"] for row in prompts)),
        "text_embedding_checkpoint_sha256": sha256_file(checkpoint),
        "ranking_contract": {
            "relevance_inputs": "CLaMP3 text/audio embeddings only",
            "metadata_use": "labels and identity-crowding diagnostics only",
            "tie_break": "track id in this frozen snapshot",
        },
    }
    atomic_json(args.output / "manifest.json", manifest)
    atomic_json(args.output / "usage.json", usage)
    atomic_json(args.output / "depth-and-cutoffs.json", depth)
    atomic_json(args.output / "composition.json", compositions)
    atomic_json(args.output / "query-pairs.json", query_pairs)
    qualitative = qualitative_markdown(usage, depth, compositions)
    atomic_text(args.output / "qualitative.md", qualitative)
    print(f"Complete: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
