#!/usr/bin/env python3
"""Evaluate Find Music composition on the exact current active phone domain.

This host-only experiment ranks existing CLaMP3 embeddings. Track metadata is emitted
only for labels and diagnostics; it never enters a relevance score. Playback ordering is
outside this evaluator's scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import v2_queue_eval as queue_eval


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DB = queue_eval.DEFAULT_DB
DEFAULT_ACCEPTANCE = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-realistic-text-battery"
)
DEFAULT_ACTIVE_CATALOG = DEFAULT_ACCEPTANCE / "active-catalog.tsv"
DEFAULT_PHONE_REPORT = DEFAULT_ACCEPTANCE / "report.json"
DEFAULT_COHORT_REPORT = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-active-domain-real-cohort"
    / "report.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "active-composition-2026-07-15"
)

EXPECTED_DB_SHA256 = queue_eval.EXPECTED_DB_SHA256
EXPECTED_ACTIVE_CATALOG_SHA256 = (
    "e5bd6f7e0e29e25ae001b83bf20af276ce7ec06aa4f53c2d7ad4e5d0a9651c75"
)
EXPECTED_ACTIVE_COUNT = 80_323
EXPECTED_QUARANTINED_COUNT = 98
TOP_K = 30
DIAGNOSTIC_K = 100
REFINE_WIDTHS: tuple[float, ...] = (0.0025, 0.005, 0.01)
OPERATORS: tuple[str, ...] = ("all_of", "either", "direction", "strict_all")
DIAGNOSTIC_AGGREGATORS: tuple[str, ...] = (
    "arithmetic_mean",
    "harmonic_mean",
    "rrf_k60",
)
RRF_K = 60
PARETO_3D_MAX_FRONTS = 12
CALIBRATED_DIRECTION = "direction_std_calibrated"


# Frozen before this evaluator inspects composed results. Text references are exact queries
# embedded in the realistic battery. Song references are seeds from the real-cohort run,
# whose active-catalog TSV is byte-identical to the realistic battery's TSV.
CASES: tuple[dict[str, object], ...] = (
    {
        "id": "text_slow_plus_psychedelic",
        "intent": "independent text ingredients versus the frozen compound phrase",
        "anchors": (("text:slow", 0.5), ("text:psychedelic", 0.5)),
        "compound_controls": ("slow psychedelic",),
    },
    {
        "id": "text_psychedelic_plus_guitar",
        "intent": "independent text ingredients versus the frozen compound phrase",
        "anchors": (("text:psychedelic", 0.5), ("text:guitar", 0.5)),
        "compound_controls": ("psychedelic guitar",),
    },
    {
        "id": "text_slow_psychedelic_guitar",
        "intent": "three independently weighted text ingredients",
        "anchors": (
            ("text:slow", 1.0 / 3.0),
            ("text:psychedelic", 1.0 / 3.0),
            ("text:guitar", 1.0 / 3.0),
        ),
        "compound_controls": ("slow psychedelic", "psychedelic guitar"),
        "missing_exact_compound": "slow psychedelic guitar",
    },
    {
        "id": "text_ambient_plus_psychedelic",
        "intent": "cross-concept text conjunction/union",
        "anchors": (("text:ambient", 0.5), ("text:psychedelic", 0.5)),
        "compound_controls": (),
    },
    {
        "id": "text_relaxing_plus_guitar",
        "intent": "independent text ingredients versus the frozen compound phrase",
        "anchors": (("text:relaxing", 0.5), ("text:guitar", 0.5)),
        "compound_controls": ("relaxing guitar",),
    },
    {
        "id": "text_sleep_plus_ambient",
        "intent": "two text ingredients versus a nearby natural-language phrase",
        "anchors": (("text:sleep", 0.5), ("text:ambient", 0.5)),
        "compound_controls": ("music for falling asleep",),
    },
    {
        "id": "text_ambient_avoid_psychedelic",
        "intent": "positive text request with an Avoid ingredient",
        "anchors": (("text:ambient", 0.75), ("text:psychedelic", -0.25)),
        "compound_controls": (),
    },
    {
        "id": "song_bonobo_with_ambient",
        "intent": "current-cohort song anchored by a text characteristic",
        "anchors": (("song:80437", 0.65), ("text:ambient", 0.35)),
        "compound_controls": (),
    },
    {
        "id": "song_neroche_with_sleep",
        "intent": "current-cohort song anchored by a text characteristic",
        "anchors": (("song:42335", 0.65), ("text:sleep", 0.35)),
        "compound_controls": (),
    },
    {
        "id": "song_kailash_with_sufi",
        "intent": "current-cohort song anchored by a text characteristic",
        "anchors": (("song:33821", 0.65), ("text:sufi devotional music", 0.35)),
        "compound_controls": (),
    },
    {
        "id": "song_khruangbin_with_desert_blues",
        "intent": "current-cohort song anchored by a text characteristic",
        "anchors": (("song:5987", 0.65), ("text:desert blues", 0.35)),
        "compound_controls": (),
    },
    {
        "id": "song_bonobo_or_kailash",
        "intent": "two current-cohort song alternatives",
        "anchors": (("song:80437", 0.5), ("song:33821", 0.5)),
        "compound_controls": (),
    },
    {
        "id": "song_kasabian_avoid_guitar",
        "intent": "current-cohort song with an Avoid text ingredient",
        "anchors": (("song:38327", 0.75), ("text:guitar", -0.25)),
        "compound_controls": (),
    },
)


REFINE_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "refine_bonobo_with_ambient",
        "primary": "song:80437",
        "secondary": "text:ambient",
        "secondary_sign": 1,
    },
    {
        "id": "refine_kailash_with_sufi",
        "primary": "song:33821",
        "secondary": "text:sufi devotional music",
        "secondary_sign": 1,
    },
    {
        "id": "refine_khruangbin_with_desert_blues",
        "primary": "song:5987",
        "secondary": "text:desert blues",
        "secondary_sign": 1,
    },
    {
        "id": "refine_slow_with_psychedelic",
        "primary": "text:slow",
        "secondary": "text:psychedelic",
        "secondary_sign": 1,
    },
    {
        "id": "refine_ambient_away_psychedelic",
        "primary": "text:ambient",
        "secondary": "text:psychedelic",
        "secondary_sign": -1,
    },
    {
        "id": "refine_kasabian_away_guitar",
        "primary": "song:38327",
        "secondary": "text:guitar",
        "secondary_sign": -1,
    },
)


WEIGHT_SWEEPS: tuple[dict[str, object], ...] = (
    {
        "id": "weight_slow_psychedelic",
        "anchors": ("text:slow", "text:psychedelic"),
    },
    {
        "id": "weight_relaxing_guitar",
        "anchors": ("text:relaxing", "text:guitar"),
    },
    {
        "id": "weight_sleep_ambient",
        "anchors": ("text:sleep", "text:ambient"),
    },
    {
        "id": "weight_bonobo_ambient",
        "anchors": ("song:80437", "text:ambient"),
    },
    {
        "id": "weight_kailash_sufi",
        "anchors": ("song:33821", "text:sufi devotional music"),
    },
)


@dataclass(frozen=True)
class ActiveCatalog:
    schema: str
    database_generation: str
    provider_generation: str
    active_track_ids: np.ndarray
    quarantined_track_ids: np.ndarray
    poweramp_file_ids: dict[int, int]
    binding_evidence: dict[int, str]
    unbound_poweramp_file_ids: tuple[int, ...]


@dataclass(frozen=True)
class ActiveLibrary:
    track_ids: np.ndarray
    embeddings: np.ndarray
    artists: tuple[str | None, ...]
    albums: tuple[str | None, ...]
    titles: tuple[str | None, ...]
    durations_ms: np.ndarray
    file_paths: tuple[str, ...]
    poweramp_file_ids: np.ndarray
    binding_evidence: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.track_ids.size)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class Anchor:
    reference: str
    label: str
    embedding: np.ndarray
    weight: float

    @property
    def negative(self) -> bool:
        return self.weight < 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_active_catalog(path: Path) -> ActiveCatalog:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError("active catalog is incomplete")
    if sha256_file(path) != EXPECTED_ACTIVE_CATALOG_SHA256:
        raise ValueError("active catalog SHA-256 does not match the frozen battery")
    schema = lines[0]
    database_generation = lines[1].split("\t", 1)[1]
    provider_generation = lines[2].split("\t", 1)[1]
    counts = lines[3].split("\t")
    declared = {counts[i]: int(counts[i + 1]) for i in range(1, len(counts), 2)}

    active: list[int] = []
    quarantined: list[int] = []
    poweramp_file_ids: dict[int, int] = {}
    binding_evidence: dict[int, str] = {}
    unbound: list[int] = []
    for line_number, line in enumerate(lines[4:], start=5):
        parts = line.split("\t")
        if parts[0] == "ACTIVE" and len(parts) == 4:
            track_id = int(parts[1])
            active.append(track_id)
            poweramp_file_ids[track_id] = int(parts[2])
            binding_evidence[track_id] = parts[3]
        elif parts[0] == "QUARANTINED" and len(parts) == 3:
            quarantined.append(int(parts[1]))
        elif parts[0] == "UNBOUND_POWERAMP" and len(parts) == 2:
            unbound.append(int(parts[1]))
        else:
            raise ValueError(f"invalid active catalog row {line_number}: {parts}")

    if active != sorted(active) or len(active) != len(set(active)):
        raise ValueError("active track IDs are not strictly ordered")
    if quarantined != sorted(quarantined) or len(quarantined) != len(set(quarantined)):
        raise ValueError("quarantined track IDs are not strictly ordered")
    if set(active) & set(quarantined):
        raise ValueError("active and quarantine partitions overlap")
    if len(active) != declared["active"] or len(active) != EXPECTED_ACTIVE_COUNT:
        raise ValueError("active count does not match the frozen contract")
    if len(quarantined) != declared["quarantined"] or len(quarantined) != EXPECTED_QUARANTINED_COUNT:
        raise ValueError("quarantine count does not match the frozen contract")
    if len(unbound) != declared["unbound_provider"]:
        raise ValueError("unbound provider count does not match the frozen contract")
    if len(active) + len(quarantined) != declared["database"]:
        raise ValueError("active/quarantine rows do not partition the database")

    return ActiveCatalog(
        schema=schema,
        database_generation=database_generation,
        provider_generation=provider_generation,
        active_track_ids=np.asarray(active, dtype=np.int64),
        quarantined_track_ids=np.asarray(quarantined, dtype=np.int64),
        poweramp_file_ids=poweramp_file_ids,
        binding_evidence=binding_evidence,
        unbound_poweramp_file_ids=tuple(unbound),
    )


def load_active_library(
    db_path: Path,
    active_catalog: ActiveCatalog,
    verify_hash: bool,
) -> tuple[ActiveLibrary, str]:
    full, db_hash = queue_eval.load_library(db_path, verify_hash=verify_hash)
    all_partition_ids = np.sort(
        np.concatenate(
            (active_catalog.active_track_ids, active_catalog.quarantined_track_ids)
        )
    )
    if not np.array_equal(full.track_ids, all_partition_ids):
        raise ValueError("active catalog does not exactly partition the frozen host DB")
    source_indices = np.searchsorted(full.track_ids, active_catalog.active_track_ids)
    if not np.array_equal(full.track_ids[source_indices], active_catalog.active_track_ids):
        raise ValueError("active track projection is not one-to-one")

    library = ActiveLibrary(
        track_ids=full.track_ids[source_indices].copy(),
        embeddings=full.embeddings[source_indices].copy(),
        artists=tuple(full.artists[int(index)] for index in source_indices),
        albums=tuple(full.albums[int(index)] for index in source_indices),
        titles=tuple(full.titles[int(index)] for index in source_indices),
        durations_ms=full.durations_ms[source_indices].copy(),
        file_paths=tuple(full.file_paths[int(index)] for index in source_indices),
        poweramp_file_ids=np.asarray(
            [active_catalog.poweramp_file_ids[int(track_id)] for track_id in active_catalog.active_track_ids],
            dtype=np.int64,
        ),
        binding_evidence=tuple(
            active_catalog.binding_evidence[int(track_id)]
            for track_id in active_catalog.active_track_ids
        ),
    )
    if library.count != EXPECTED_ACTIVE_COUNT or library.dim != queue_eval.EXPECTED_DIM:
        raise ValueError("active embedding matrix shape is wrong")
    return library, db_hash


def load_phone_text_embeddings(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, object]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in report["textRuns"]:
        grouped.setdefault(str(row["query"]), []).append(row)

    embeddings: dict[str, np.ndarray] = {}
    declared_hashes: dict[str, str] = {}
    for query, rows in grouped.items():
        if sorted(int(row["repeat"]) for row in rows) != [1, 2]:
            raise ValueError(f"query {query!r} does not have exactly repeats 1 and 2")
        vectors = [np.asarray(row["embedding"], dtype="<f4") for row in rows]
        if any(vector.shape != (queue_eval.EXPECTED_DIM,) for vector in vectors):
            raise ValueError(f"query {query!r} has an invalid embedding shape")
        if not np.array_equal(vectors[0], vectors[1]):
            raise ValueError(f"query {query!r} changed between phone repeats")
        hashes = {str(row["embeddingSha256"]) for row in rows}
        if len(hashes) != 1:
            raise ValueError(f"query {query!r} changed hash between phone repeats")
        declared_hash = hashes.pop()
        actual_hash = hashlib.sha256(vectors[0].tobytes()).hexdigest()
        if actual_hash != declared_hash:
            raise ValueError(f"query {query!r} embedding bytes do not match the phone hash")
        embeddings[query] = vectors[0]
        declared_hashes[query] = declared_hash
    return embeddings, declared_hashes, report


def empirical_percentiles(similarities: np.ndarray) -> np.ndarray:
    """Production R3 upper-CDF percentiles: exactly equal Float32 scores stay tied."""
    values = np.asarray(similarities, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("percentiles require a non-empty finite Float32 vector")
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    group_end = np.flatnonzero(
        np.concatenate((ordered[1:] != ordered[:-1], np.asarray([True])))
    )
    counts = np.diff(np.concatenate((np.asarray([-1]), group_end)))
    group_values = ((group_end + 1) / values.size).astype(np.float32)
    ranks = np.empty(values.size, dtype=np.float32)
    ranks[order] = np.repeat(group_values, counts)
    return ranks


class AnchorCache:
    def __init__(
        self,
        library: ActiveLibrary,
        text_embeddings: dict[str, np.ndarray],
    ) -> None:
        self.library = library
        self.text_embeddings = text_embeddings
        self._track_position = {
            int(track_id): index for index, track_id in enumerate(library.track_ids)
        }
        self._similarities: dict[str, np.ndarray] = {}
        self._positive_percentiles: dict[str, np.ndarray] = {}
        self._negative_percentiles: dict[str, np.ndarray] = {}

    def embedding(self, reference: str) -> np.ndarray:
        kind, value = reference.split(":", 1)
        if kind == "text":
            return self.text_embeddings[value]
        if kind == "song":
            return self.library.embeddings[self._track_position[int(value)]]
        raise ValueError(f"unknown anchor reference: {reference}")

    def label(self, reference: str) -> str:
        kind, value = reference.split(":", 1)
        if kind == "text":
            return value
        index = self._track_position[int(value)]
        return f"{self.library.artists[index] or '?'} - {self.library.titles[index] or '?'}"

    def similarities(self, reference: str) -> np.ndarray:
        result = self._similarities.get(reference)
        if result is None:
            result = np.asarray(
                self.library.embeddings @ self.embedding(reference),
                dtype=np.float32,
            )
            self._similarities[reference] = result
        return result

    def percentiles(self, reference: str, negative: bool) -> np.ndarray:
        cache = self._negative_percentiles if negative else self._positive_percentiles
        result = cache.get(reference)
        if result is None:
            values = -self.similarities(reference) if negative else self.similarities(reference)
            result = empirical_percentiles(values)
            cache[reference] = result
        return result

    def resolve(self, spec: tuple[str, float]) -> Anchor:
        reference, weight = spec
        return Anchor(reference, self.label(reference), self.embedding(reference), float(weight))


def all_of_scores(percentiles: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    total = sum(abs(float(weight)) for weight in weights)
    if total <= 1e-8:
        raise ValueError("All of needs at least one non-zero ingredient")
    logs = np.zeros(percentiles[0].size, dtype=np.float64)
    for values, weight in zip(percentiles, weights):
        normalized = abs(float(weight)) / total
        logs += normalized * np.log(
            np.clip(values.astype(np.float64), 1.0 / values.size, 1.0)
        )
    return np.exp(logs).astype(np.float32)


def normalized_signed_centroid(anchors: Sequence[Anchor]) -> np.ndarray | None:
    active = [anchor for anchor in anchors if abs(anchor.weight) > 1e-12]
    total = sum(abs(anchor.weight) for anchor in active)
    if total <= 1e-12:
        return None
    vector = np.zeros(active[0].embedding.size, dtype=np.float64)
    for anchor in active:
        vector += anchor.embedding.astype(np.float64) * (anchor.weight / total)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-7:
        return None
    return (vector / norm).astype(np.float32)


def normalized_variance_calibrated_centroid(
    anchors: Sequence[Anchor],
    similarity_stds: Sequence[float],
) -> np.ndarray | None:
    """One vector whose dot product ranks a weighted sum of standardized cosines."""
    if len(anchors) != len(similarity_stds):
        raise ValueError("one active-domain cosine standard deviation is required per anchor")
    active = [
        (anchor, float(std))
        for anchor, std in zip(anchors, similarity_stds)
        if abs(anchor.weight) > 1e-12
    ]
    if not active or any(not math.isfinite(std) or std <= 1e-8 for _, std in active):
        return None
    vector = np.zeros(active[0][0].embedding.size, dtype=np.float64)
    for anchor, std in active:
        vector += anchor.embedding.astype(np.float64) * (anchor.weight / std)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-7:
        return None
    return (vector / norm).astype(np.float32)


def eligible_indices(track_ids: np.ndarray, excluded_ids: set[int]) -> np.ndarray:
    if not excluded_ids:
        return np.arange(track_ids.size, dtype=np.int64)
    excluded = np.fromiter(sorted(excluded_ids), dtype=np.int64)
    return np.flatnonzero(~np.isin(track_ids, excluded))


def rank_scalar(
    track_ids: np.ndarray,
    scores: np.ndarray,
    excluded_ids: set[int],
    count: int,
    tie_scores: np.ndarray | None = None,
) -> np.ndarray:
    eligible = eligible_indices(track_ids, excluded_ids)
    if tie_scores is None:
        order = np.lexsort((track_ids[eligible], -scores[eligible]))
    else:
        order = np.lexsort(
            (track_ids[eligible], -tie_scores[eligible], -scores[eligible])
        )
    return eligible[order[:count]]


def rank_either(
    track_ids: np.ndarray,
    percentiles: Sequence[np.ndarray],
    weights: Sequence[float],
    excluded_ids: set[int],
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if any(weight < 0 for weight in weights):
        raise ValueError("Either does not accept Avoid ingredients")
    active_anchors = [index for index, weight in enumerate(weights) if weight > 0]
    if not active_anchors:
        raise ValueError("Either needs at least one positive ingredient")
    total = sum(weights[index] for index in active_anchors)
    shares = np.asarray([weights[index] / total for index in active_anchors], dtype=np.float64)
    orders = [
        np.lexsort((track_ids, -percentiles[anchor])) for anchor in active_anchors
    ]
    cursors = np.zeros(len(active_anchors), dtype=np.int64)
    allocations = np.zeros(len(active_anchors), dtype=np.int64)
    selected_ids: set[int] = set()
    selected: list[int] = []
    winners: list[int] = []
    while len(selected) < count:
        target = len(selected) + 1
        deficits = shares * target - allocations
        branch_priority = sorted(
            range(len(active_anchors)), key=lambda position: (-deficits[position], position)
        )
        picked = False
        for branch_position in branch_priority:
            order = orders[branch_position]
            while cursors[branch_position] < order.size:
                index = int(order[cursors[branch_position]])
                cursors[branch_position] += 1
                track_id = int(track_ids[index])
                if track_id in excluded_ids or track_id in selected_ids:
                    continue
                selected_ids.add(track_id)
                selected.append(index)
                winners.append(active_anchors[branch_position])
                allocations[branch_position] += 1
                picked = True
                break
            if picked:
                break
        if not picked:
            break
    return np.asarray(selected, dtype=np.int64), np.asarray(winners, dtype=np.int64)


def excluded_song_ids(anchors: Sequence[Anchor]) -> set[int]:
    return {
        int(anchor.reference.split(":", 1)[1])
        for anchor in anchors
        if anchor.reference.startswith("song:") and abs(anchor.weight) > 1e-12
    }


def anchor_records(anchors: Sequence[Anchor]) -> list[dict[str, object]]:
    return [
        {
            "reference": anchor.reference,
            "label": anchor.label,
            "weight": anchor.weight,
            "negative": anchor.negative,
        }
        for anchor in anchors
    ]


def row_record(
    library: ActiveLibrary,
    index: int,
    rank: int,
    objective_score: float,
    similarities: Sequence[np.ndarray],
    percentiles: Sequence[np.ndarray],
    winning_anchor_index: int | None = None,
    objective_tie_score: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "rank": rank,
        "track_id": int(library.track_ids[index]),
        "poweramp_file_id": int(library.poweramp_file_ids[index]),
        "binding_evidence": library.binding_evidence[index],
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "objective_score": float(objective_score),
        "anchor_cosines": [float(values[index]) for values in similarities],
        "effective_anchor_percentiles": [
            float(values[index]) for values in percentiles
        ],
    }
    if winning_anchor_index is not None:
        row["winning_anchor_index"] = winning_anchor_index
    if objective_tie_score is not None:
        row["objective_tie_score"] = objective_tie_score
    return row


def list_diagnostics(
    library: ActiveLibrary,
    selected: np.ndarray,
    percentiles: Sequence[np.ndarray],
) -> dict[str, object]:
    top = selected[:TOP_K]
    evidence = np.stack([values[top] for values in percentiles], axis=1)
    row_worst = np.min(evidence, axis=1)
    threshold = 1.0 - (100.0 / library.count)
    artists = {
        (library.artists[int(index)] or "").casefold()
        for index in top
        if library.artists[int(index)]
    }
    return {
        "mean_effective_percentile_by_anchor": np.mean(evidence, axis=0).tolist(),
        "min_effective_percentile_by_anchor": np.min(evidence, axis=0).tolist(),
        "mean_row_worst_anchor_percentile": float(np.mean(row_worst)),
        "min_row_worst_anchor_percentile": float(np.min(row_worst)),
        "anchor_top100_coverage": [
            int(np.sum(values[top] > threshold)) for values in percentiles
        ],
        "distinct_display_artist_count": len(artists),
        "top30_track_id_sha256": hashlib.sha256(
            ",".join(str(int(library.track_ids[index])) for index in top).encode("ascii")
        ).hexdigest(),
    }


def evaluate_operator(
    library: ActiveLibrary,
    cache: AnchorCache,
    anchors: Sequence[Anchor],
    operator: str,
) -> dict[str, object]:
    active = [anchor for anchor in anchors if abs(anchor.weight) > 1e-12]
    similarities = [cache.similarities(anchor.reference) for anchor in active]
    percentiles = [
        cache.percentiles(anchor.reference, anchor.negative) for anchor in active
    ]
    weights = [anchor.weight for anchor in active]
    excluded = excluded_song_ids(active)
    selected: np.ndarray
    winning: np.ndarray | None = None
    objective: np.ndarray | None = None
    tie_objective: np.ndarray | None = None
    direction_vector_sha256: str | None = None
    direction_calibration_stds: list[float] | None = None

    try:
        if operator == "all_of":
            objective = all_of_scores(percentiles, weights)
            selected = rank_scalar(
                library.track_ids, objective, excluded, DIAGNOSTIC_K
            )
        elif operator == "either":
            selected, winning = rank_either(
                library.track_ids,
                percentiles,
                weights,
                excluded,
                DIAGNOSTIC_K,
            )
        elif operator in {"direction", CALIBRATED_DIRECTION}:
            if operator == CALIBRATED_DIRECTION:
                direction_calibration_stds = [
                    float(np.std(values, dtype=np.float64))
                    for values in similarities
                ]
                direction = normalized_variance_calibrated_centroid(
                    active, direction_calibration_stds
                )
            else:
                direction = normalized_signed_centroid(active)
            if direction is None:
                raise ValueError("signed ingredients cancel or have invalid calibration")
            direction_vector_sha256 = hashlib.sha256(
                direction.astype("<f4").tobytes()
            ).hexdigest()
            objective = np.asarray(library.embeddings @ direction, dtype=np.float32)
            selected = rank_scalar(
                library.track_ids, objective, excluded, DIAGNOSTIC_K
            )
        elif operator == "strict_all":
            objective = np.min(np.stack(percentiles, axis=0), axis=0).astype(np.float32)
            tie_objective = all_of_scores(percentiles, weights)
            selected = rank_scalar(
                library.track_ids,
                objective,
                excluded,
                DIAGNOSTIC_K,
                tie_scores=tie_objective,
            )
        else:
            raise ValueError(f"unknown operator: {operator}")
    except ValueError as error:
        return {
            "operator": operator,
            "status": "rejected",
            "reason": str(error),
            "anchors": anchor_records(active),
        }

    rows: list[dict[str, object]] = []
    for rank, index in enumerate(selected[:TOP_K], start=1):
        winner = int(winning[rank - 1]) if winning is not None else None
        if objective is not None:
            score = float(objective[index])
        else:
            assert winner is not None
            score = float(percentiles[winner][index])
        rows.append(
            row_record(
                library=library,
                index=int(index),
                rank=rank,
                objective_score=score,
                similarities=similarities,
                percentiles=percentiles,
                winning_anchor_index=winner,
                objective_tie_score=(
                    float(tie_objective[index]) if tie_objective is not None else None
                ),
            )
        )

    result: dict[str, object] = {
        "operator": operator,
        "status": "ranked",
        "anchors": anchor_records(active),
        "diagnostics": list_diagnostics(library, selected, percentiles),
        "top30": rows,
        "top100_track_ids": [
            int(library.track_ids[index]) for index in selected[:DIAGNOSTIC_K]
        ],
    }
    if winning is not None:
        result["top30_branch_allocations"] = [
            int(np.sum(winning[:TOP_K] == index)) for index in range(len(active))
        ]
        result["top30_winning_anchor_sequence"] = winning[:TOP_K].tolist()
    if direction_vector_sha256 is not None:
        result["normalized_signed_centroid_sha256"] = direction_vector_sha256
    if direction_calibration_stds is not None:
        result["active_domain_anchor_cosine_stds"] = direction_calibration_stds
        result["calibration_contract"] = (
            "scale each signed anchor vector by inverse active-domain cosine standard "
            "deviation before final normalization; cosine centering is rank-constant"
        )
    return result


def diagnostic_aggregate_scores(
    aggregator: str,
    percentiles: Sequence[np.ndarray],
    weights: Sequence[float],
) -> np.ndarray:
    if any(weight <= 0 for weight in weights):
        raise ValueError("diagnostic aggregators require positive ingredients")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= np.sum(normalized)
    evidence = np.stack(percentiles, axis=0).astype(np.float64)
    if aggregator == "arithmetic_mean":
        return np.sum(evidence * normalized[:, None], axis=0).astype(np.float32)
    if aggregator == "harmonic_mean":
        return (1.0 / np.sum(normalized[:, None] / evidence, axis=0)).astype(np.float32)
    if aggregator == "rrf_k60":
        size = evidence.shape[1]
        # A production percentile m/N has exactly N-m strictly better rows.
        ranks = 1 + size - np.rint(evidence * size).astype(np.int64)
        return np.sum(
            normalized[:, None] / (RRF_K + ranks), axis=0
        ).astype(np.float32)
    raise ValueError(f"unknown diagnostic aggregator: {aggregator}")


def evaluate_diagnostic_aggregator(
    library: ActiveLibrary,
    cache: AnchorCache,
    anchors: Sequence[Anchor],
    aggregator: str,
) -> dict[str, object]:
    active = [anchor for anchor in anchors if abs(anchor.weight) > 1e-12]
    if len(active) < 2 or any(anchor.weight <= 0 for anchor in active):
        return {
            "aggregator": aggregator,
            "status": "not_applicable",
            "reason": "diagnostic applies only to two or three positive ingredients",
        }
    similarities = [cache.similarities(anchor.reference) for anchor in active]
    percentiles = [cache.percentiles(anchor.reference, False) for anchor in active]
    weights = [anchor.weight for anchor in active]
    scores = diagnostic_aggregate_scores(aggregator, percentiles, weights)
    selected = rank_scalar(
        library.track_ids,
        scores,
        excluded_song_ids(active),
        DIAGNOSTIC_K,
    )
    return {
        "aggregator": aggregator,
        "status": "ranked",
        "contract": (
            "diagnostic total order only; not a proposed UI mode or playback ordering policy"
        ),
        "anchors": anchor_records(active),
        "diagnostics": list_diagnostics(library, selected, percentiles),
        "top30": [
            row_record(
                library,
                int(index),
                rank,
                float(scores[index]),
                similarities,
                percentiles,
            )
            for rank, index in enumerate(selected[:TOP_K], start=1)
        ],
        "top100_track_ids": [
            int(library.track_ids[index]) for index in selected[:DIAGNOSTIC_K]
        ],
    }


def pareto_depths_2d(percentiles: Sequence[np.ndarray]) -> np.ndarray:
    """Exact non-dominated sorting depth for two maximize objectives in O(N log N)."""
    first = np.asarray(percentiles[0], dtype=np.float32)
    second = np.asarray(percentiles[1], dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("Pareto objectives must align")
    order = np.lexsort((-second, -first))
    unique_second = np.unique(second)[::-1]
    coordinates = np.searchsorted(-unique_second, -second)
    tree = np.zeros(unique_second.size + 1, dtype=np.int32)
    depths = np.zeros(first.size, dtype=np.int32)

    def query(end_inclusive: int) -> int:
        position = end_inclusive + 1
        result = 0
        while position > 0:
            result = max(result, int(tree[position]))
            position -= position & -position
        return result

    def update(coordinate: int, value: int) -> None:
        position = coordinate + 1
        while position < tree.size:
            if value > tree[position]:
                tree[position] = value
            position += position & -position

    position = 0
    while position < order.size:
        end = position + 1
        index = int(order[position])
        while end < order.size:
            other = int(order[end])
            if first[other] != first[index] or second[other] != second[index]:
                break
            end += 1
        depth = query(int(coordinates[index])) + 1
        depths[order[position:end]] = depth
        update(int(coordinates[index]), depth)
        position = end
    return depths


def nondominated_mask_3d(values: np.ndarray) -> np.ndarray:
    """Exact first-front mask for three maximize objectives in O(N log N)."""
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("expected an N x 3 Pareto matrix")
    first, second, third = values.T
    order = np.lexsort((-third, -second, -first))
    unique_second = np.unique(second)[::-1]
    coordinates = np.searchsorted(-unique_second, -second)
    tree = np.full(unique_second.size + 1, -np.inf, dtype=np.float32)
    front = np.zeros(values.shape[0], dtype=bool)

    def query(end_inclusive: int) -> float:
        position = end_inclusive + 1
        result = -np.inf
        while position > 0:
            result = max(result, float(tree[position]))
            position -= position & -position
        return result

    def update(coordinate: int, value: float) -> None:
        position = coordinate + 1
        while position < tree.size:
            if value > tree[position]:
                tree[position] = value
            position += position & -position

    position = 0
    while position < order.size:
        end = position + 1
        index = int(order[position])
        while end < order.size and np.array_equal(values[order[end]], values[index]):
            end += 1
        is_front = query(int(coordinates[index])) < float(third[index])
        front[order[position:end]] = is_front
        update(int(coordinates[index]), float(third[index]))
        position = end
    return front


def pareto_diagnostics(
    library: ActiveLibrary,
    cache: AnchorCache,
    anchors: Sequence[Anchor],
    core_operators: Sequence[dict[str, object]],
) -> dict[str, object]:
    active = [anchor for anchor in anchors if abs(anchor.weight) > 1e-12]
    if len(active) not in (2, 3) or any(anchor.weight <= 0 for anchor in active):
        return {
            "status": "not_applicable",
            "reason": "Pareto diagnostics require two or three positive ingredients",
        }
    evidence = [cache.percentiles(anchor.reference, False) for anchor in active]
    matrix = np.stack(evidence, axis=1)
    if len(active) == 2:
        depths = pareto_depths_2d(evidence)
        counts = np.bincount(depths)[1:]
        summary: dict[str, object] = {
            "status": "exact",
            "dimensions": 2,
            "front_1_size": int(counts[0]),
            "max_depth": int(depths.max()),
            "first_20_front_sizes": counts[:20].tolist(),
        }
    else:
        remaining = np.arange(library.count, dtype=np.int64)
        depths = np.zeros(library.count, dtype=np.int32)
        sizes: list[int] = []
        for depth in range(1, PARETO_3D_MAX_FRONTS + 1):
            mask = nondominated_mask_3d(matrix[remaining])
            front = remaining[mask]
            if front.size == 0:
                break
            depths[front] = depth
            sizes.append(int(front.size))
            remaining = remaining[~mask]
            if remaining.size == 0:
                break
        summary = {
            "status": "fronts_capped",
            "dimensions": 3,
            "computed_fronts": len(sizes),
            "front_sizes": sizes,
            "remaining_after_computed_fronts": int(remaining.size),
        }

    position_by_id = {
        int(track_id): position for position, track_id in enumerate(library.track_ids)
    }
    selected_depths: list[dict[str, object]] = []
    for record in core_operators:
        if record["status"] != "ranked" or record["operator"] not in {"all_of", "strict_all"}:
            continue
        positions = [position_by_id[int(row["track_id"])] for row in record["top30"]]
        values = depths[positions]
        selected_depths.append(
            {
                "operator": record["operator"],
                "top30_depths": values.tolist(),
                "mean_computed_depth": float(np.mean(values)),
                "max_computed_depth": int(np.max(values)),
                "beyond_computed_fronts": int(np.sum(values == 0)),
            }
        )
    summary["core_top30_depths"] = selected_depths
    summary["interpretation"] = (
        "Pareto fronts diagnose trade-off multiplicity but do not define a deterministic "
        "total order suitable for a result list."
    )
    return summary


def direct_text_control(
    library: ActiveLibrary,
    cache: AnchorCache,
    query: str,
) -> dict[str, object]:
    reference = f"text:{query}"
    similarities = cache.similarities(reference)
    percentiles = [cache.percentiles(reference, False)]
    selected = rank_scalar(library.track_ids, similarities, set(), DIAGNOSTIC_K)
    return {
        "query": query,
        "embedding_sha256": hashlib.sha256(
            cache.embedding(reference).astype("<f4").tobytes()
        ).hexdigest(),
        "top30": [
            row_record(
                library,
                int(index),
                rank,
                float(similarities[index]),
                [similarities],
                percentiles,
            )
            for rank, index in enumerate(selected[:TOP_K], start=1)
        ],
        "top100_track_ids": [int(library.track_ids[index]) for index in selected],
    }


def compound_embedding_geometry(
    cache: AnchorCache,
    anchors: Sequence[Anchor],
    queries: Sequence[str],
) -> list[dict[str, object]]:
    if not queries or any(anchor.weight <= 0 for anchor in anchors):
        return []
    direction = normalized_signed_centroid(anchors)
    if direction is None:
        return []
    return [
        {
            "compound_query": query,
            "cosine_to_normalized_weighted_anchor_centroid": float(
                np.dot(cache.embedding(f"text:{query}"), direction)
            ),
            "cosines_to_individual_anchors": [
                float(np.dot(cache.embedding(f"text:{query}"), anchor.embedding))
                for anchor in anchors
            ],
        }
        for query in queries
    ]


def overlap_record(first: Sequence[int], second: Sequence[int]) -> dict[str, object]:
    first_set = set(first)
    second_set = set(second)
    intersection = len(first_set & second_set)
    union = len(first_set | second_set)
    same_position = sum(a == b for a, b in zip(first, second))
    return {
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
        "same_position": same_position,
    }


def attach_case_overlaps(
    case_record: dict[str, object],
    controls: dict[str, dict[str, object]],
) -> None:
    ranked = {
        str(record["operator"]): record
        for record in case_record["operators"]
        if record["status"] == "ranked"
    }
    pairwise: list[dict[str, object]] = []
    names = sorted(ranked)
    for first_position, first in enumerate(names):
        for second in names[first_position + 1 :]:
            first_ids = [int(row["track_id"]) for row in ranked[first]["top30"]]
            second_ids = [int(row["track_id"]) for row in ranked[second]["top30"]]
            pairwise.append(
                {
                    "first": first,
                    "second": second,
                    "top30": overlap_record(first_ids, second_ids),
                    "top100": overlap_record(
                        ranked[first]["top100_track_ids"],
                        ranked[second]["top100_track_ids"],
                    ),
                }
            )
    case_record["pairwise_operator_overlap"] = pairwise

    control_comparisons: list[dict[str, object]] = []
    for query in case_record["compound_controls"]:
        control = controls[str(query)]
        control_top30 = [int(row["track_id"]) for row in control["top30"]]
        for operator, record in ranked.items():
            result_top30 = [int(row["track_id"]) for row in record["top30"]]
            control_comparisons.append(
                {
                    "compound_query": query,
                    "operator": operator,
                    "top30": overlap_record(result_top30, control_top30),
                    "top100": overlap_record(
                        record["top100_track_ids"], control["top100_track_ids"]
                    ),
                }
            )
    case_record["compound_control_overlap"] = control_comparisons


def attach_diagnostic_overlaps(case_record: dict[str, object]) -> None:
    core = {
        str(record["operator"]): record
        for record in case_record["operators"]
        if record["status"] == "ranked" and record["operator"] in {"all_of", "strict_all"}
    }
    comparisons: list[dict[str, object]] = []
    for diagnostic in case_record["diagnostic_aggregators"]:
        if diagnostic["status"] != "ranked":
            continue
        diagnostic_top30 = [int(row["track_id"]) for row in diagnostic["top30"]]
        for core_name, core_record in core.items():
            core_top30 = [int(row["track_id"]) for row in core_record["top30"]]
            comparisons.append(
                {
                    "aggregator": diagnostic["aggregator"],
                    "core_operator": core_name,
                    "top30": overlap_record(diagnostic_top30, core_top30),
                    "top100": overlap_record(
                        diagnostic["top100_track_ids"],
                        core_record["top100_track_ids"],
                    ),
                    "aggregator_mean_row_worst_anchor_percentile": diagnostic[
                        "diagnostics"
                    ]["mean_row_worst_anchor_percentile"],
                    "core_mean_row_worst_anchor_percentile": core_record["diagnostics"][
                        "mean_row_worst_anchor_percentile"
                    ],
                }
            )
    case_record["diagnostic_overlap_with_core"] = comparisons

    calibrated_comparisons: list[dict[str, object]] = []
    for diagnostic in case_record["direction_diagnostics"]:
        if diagnostic["status"] != "ranked":
            continue
        diagnostic_ids = [int(row["track_id"]) for row in diagnostic["top30"]]
        for core_name in ("all_of", "direction", "strict_all"):
            core_record = next(
                (
                    record
                    for record in case_record["operators"]
                    if record["operator"] == core_name and record["status"] == "ranked"
                ),
                None,
            )
            if core_record is None:
                continue
            core_ids = [int(row["track_id"]) for row in core_record["top30"]]
            calibrated_comparisons.append(
                {
                    "diagnostic": diagnostic["operator"],
                    "core_operator": core_name,
                    "top30": overlap_record(diagnostic_ids, core_ids),
                    "diagnostic_mean_row_worst_anchor_percentile": diagnostic[
                        "diagnostics"
                    ]["mean_row_worst_anchor_percentile"],
                    "core_mean_row_worst_anchor_percentile": core_record["diagnostics"][
                        "mean_row_worst_anchor_percentile"
                    ],
                }
            )
    case_record["calibrated_direction_overlap_with_core"] = calibrated_comparisons


def refine_record(
    library: ActiveLibrary,
    cache: AnchorCache,
    definition: dict[str, object],
) -> dict[str, object]:
    primary_ref = str(definition["primary"])
    secondary_ref = str(definition["secondary"])
    secondary_sign = int(definition["secondary_sign"])
    primary_similarities = cache.similarities(primary_ref)
    secondary_similarities = cache.similarities(secondary_ref)
    primary_percentiles = cache.percentiles(primary_ref, False)
    secondary_percentiles = cache.percentiles(secondary_ref, secondary_sign < 0)
    excluded = {
        int(reference.split(":", 1)[1])
        for reference in (primary_ref, secondary_ref)
        if reference.startswith("song:")
    }
    raw_primary = rank_scalar(
        library.track_ids, primary_similarities, excluded, TOP_K
    )
    raw_secondary_mean = float(np.mean(secondary_percentiles[raw_primary]))
    widths: list[dict[str, object]] = []
    for fraction in REFINE_WIDTHS:
        eligible = np.flatnonzero(primary_percentiles > 1.0 - fraction)
        if excluded:
            eligible = eligible[
                ~np.isin(
                    library.track_ids[eligible],
                    np.fromiter(sorted(excluded), dtype=np.int64),
                )
            ]
        order = np.lexsort(
            (
                library.track_ids[eligible],
                -primary_percentiles[eligible],
                -secondary_percentiles[eligible],
            )
        )
        selected = eligible[order[:TOP_K]]
        if selected.size != TOP_K:
            raise AssertionError(f"Refine {definition['id']} did not fill top {TOP_K}")
        floor = float(np.min(primary_percentiles[selected]))
        if not floor > 1.0 - fraction:
            raise AssertionError(f"Refine {definition['id']} violated its primary floor")
        widths.append(
            {
                "primary_fraction": fraction,
                "candidate_count": int(eligible.size),
                "primary_percentile_floor": floor,
                "mean_primary_percentile": float(
                    np.mean(primary_percentiles[selected])
                ),
                "mean_effective_secondary_percentile": float(
                    np.mean(secondary_percentiles[selected])
                ),
                "secondary_mean_gain_over_raw_primary_top30": float(
                    np.mean(secondary_percentiles[selected]) - raw_secondary_mean
                ),
                "overlap_with_raw_primary_top30": len(
                    set(selected.tolist()) & set(raw_primary.tolist())
                ),
                "top30": [
                    row_record(
                        library,
                        int(index),
                        rank,
                        float(secondary_percentiles[index]),
                        [primary_similarities, secondary_similarities],
                        [primary_percentiles, secondary_percentiles],
                        objective_tie_score=float(primary_percentiles[index]),
                    )
                    for rank, index in enumerate(selected, start=1)
                ],
            }
        )
    return {
        "id": definition["id"],
        "contract": "rank within A's declared active-domain percentile neighborhood by B",
        "primary": {
            "reference": primary_ref,
            "label": cache.label(primary_ref),
        },
        "secondary": {
            "reference": secondary_ref,
            "label": cache.label(secondary_ref),
            "relation": "away" if secondary_sign < 0 else "with",
        },
        "raw_primary_top30_mean_effective_secondary_percentile": raw_secondary_mean,
        "raw_primary_top30_track_ids": [
            int(library.track_ids[index]) for index in raw_primary
        ],
        "widths": widths,
    }


def weight_sweep_record(
    library: ActiveLibrary,
    cache: AnchorCache,
    definition: dict[str, object],
) -> dict[str, object]:
    references = tuple(str(value) for value in definition["anchors"])
    records: list[dict[str, object]] = []
    previous_by_operator: dict[str, list[int]] = {}
    for second_tenths in range(11):
        second_weight = second_tenths / 10.0
        weights = (1.0 - second_weight, second_weight)
        anchors = [cache.resolve((reference, weight)) for reference, weight in zip(references, weights)]
        step: dict[str, object] = {
            "second_weight": second_weight,
            "weights": list(weights),
            "operators": [],
        }
        for operator in (*OPERATORS, CALIBRATED_DIRECTION):
            result = evaluate_operator(library, cache, anchors, operator)
            if result["status"] != "ranked":
                raise AssertionError(f"positive weight sweep rejected {operator}")
            track_ids = [int(row["track_id"]) for row in result["top30"]]
            previous = previous_by_operator.get(operator)
            step["operators"].append(
                {
                    "operator": operator,
                    "adjacent_top30": (
                        overlap_record(previous, track_ids) if previous is not None else None
                    ),
                    "diagnostics": result["diagnostics"],
                    "top30_track_ids": track_ids,
                    "top30_branch_allocations": result.get("top30_branch_allocations"),
                }
            )
            previous_by_operator[operator] = track_ids
        records.append(step)
    return {
        "id": definition["id"],
        "anchors": [
            {"reference": reference, "label": cache.label(reference)}
            for reference in references
        ],
        "steps": records,
    }


def phone_replay_checks(
    library: ActiveLibrary,
    cache: AnchorCache,
    report: dict[str, object],
) -> dict[str, object]:
    exact_positions = 0
    total_positions = 0
    exact_lists = 0
    max_score_delta = 0.0
    per_query: list[dict[str, object]] = []
    for phone_row in report["textRuns"][::2]:
        query = str(phone_row["query"])
        similarities = cache.similarities(f"text:{query}")
        selected = rank_scalar(library.track_ids, similarities, set(), TOP_K)
        actual_ids = [int(library.track_ids[index]) for index in selected]
        expected_ids = [int(row["trackId"]) for row in phone_row["tracks"]]
        positions = sum(a == b for a, b in zip(actual_ids, expected_ids))
        deltas = [
            abs(float(similarities[selected[position]]) - float(phone_row["tracks"][position]["score"]))
            for position in range(TOP_K)
        ]
        query_max_delta = max(deltas)
        max_score_delta = max(max_score_delta, query_max_delta)
        exact_positions += positions
        total_positions += TOP_K
        exact_lists += int(actual_ids == expected_ids)
        per_query.append(
            {
                "query": query,
                "exact_positions": positions,
                "top30_overlap": len(set(actual_ids) & set(expected_ids)),
                "max_absolute_score_delta": query_max_delta,
            }
        )
    if exact_positions != total_positions or max_score_delta > 1e-6:
        raise AssertionError("host active-domain text replay diverged from the phone report")
    return {
        "queries": len(per_query),
        "exact_top30_lists": exact_lists,
        "exact_positions": exact_positions,
        "total_positions": total_positions,
        "max_absolute_score_delta": max_score_delta,
        "per_query": per_query,
    }


def endpoint_checks(library: ActiveLibrary, cache: AnchorCache) -> dict[str, bool]:
    reference = "text:ambient"
    anchor = cache.resolve((reference, 1.0))
    expected = rank_scalar(
        library.track_ids, cache.similarities(reference), set(), TOP_K
    )
    checks: dict[str, bool] = {}
    for operator in OPERATORS:
        result = evaluate_operator(library, cache, [anchor], operator)
        actual = np.asarray(
            [
                int(np.searchsorted(library.track_ids, int(row["track_id"])))
                for row in result["top30"]
            ],
            dtype=np.int64,
        )
        checks[f"single_anchor_{operator}"] = bool(np.array_equal(expected, actual))

    first = cache.resolve(("text:ambient", 0.7))
    second = cache.resolve(("text:psychedelic", 0.3))
    scaled = [
        cache.resolve(("text:ambient", 7.0)),
        cache.resolve(("text:psychedelic", 3.0)),
    ]
    for operator in OPERATORS:
        base = evaluate_operator(library, cache, [first, second], operator)
        other = evaluate_operator(library, cache, scaled, operator)
        checks[f"positive_scale_{operator}"] = [
            row["track_id"] for row in base["top30"]
        ] == [row["track_id"] for row in other["top30"]]

    vector = cache.embedding("text:ambient")
    contradiction = [
        Anchor("synthetic:same", "same", vector, 0.5),
        Anchor("synthetic:same", "same", vector, -0.5),
    ]
    checks["direction_exact_contradiction_rejected"] = (
        normalized_signed_centroid(contradiction) is None
    )
    if not all(checks.values()):
        raise AssertionError(f"endpoint checks failed: {checks}")
    return checks


def validate_rankings(
    library: ActiveLibrary,
    cases: Sequence[dict[str, object]],
    refinements: Sequence[dict[str, object]],
) -> dict[str, object]:
    active_ids = set(int(value) for value in library.track_ids)
    checked_lists = 0
    for case in cases:
        for record in (
            *case["operators"],
            *case["diagnostic_aggregators"],
            *case["direction_diagnostics"],
        ):
            if record["status"] != "ranked":
                continue
            ids = [int(row["track_id"]) for row in record["top30"]]
            if len(ids) != TOP_K or len(ids) != len(set(ids)) or not set(ids) <= active_ids:
                name = record.get("operator", record.get("aggregator"))
                raise AssertionError(f"invalid top30 for {case['id']} / {name}")
            excluded = {
                int(anchor["reference"].split(":", 1)[1])
                for anchor in record["anchors"]
                if str(anchor["reference"]).startswith("song:")
            }
            if set(ids) & excluded:
                name = record.get("operator", record.get("aggregator"))
                raise AssertionError(f"song seed leaked into {case['id']} / {name}")
            checked_lists += 1
    for record in refinements:
        for width in record["widths"]:
            ids = [int(row["track_id"]) for row in width["top30"]]
            if len(ids) != TOP_K or len(ids) != len(set(ids)) or not set(ids) <= active_ids:
                raise AssertionError(f"invalid Refine list for {record['id']}")
            checked_lists += 1
    return {
        "checked_full_top30_lists": checked_lists,
        "all_rows_active": True,
        "all_lists_unique": True,
        "song_anchors_excluded": True,
        "refine_primary_floors_enforced": True,
    }


def track_label(row: dict[str, object]) -> str:
    return f"{row['artist'] or '?'} - {row['title'] or '?'}".replace("|", "/")


def write_listening_packet(
    path: Path,
    cases: Sequence[dict[str, object]],
    controls: dict[str, dict[str, object]],
    refinements: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# Active-Domain Composition Listening Packet",
        "",
        "All lists contain 30 active-domain rows. Metadata is a label only; every ranking",
        "was computed from CLaMP3 audio/text embeddings. Playback ordering is not evaluated.",
        "",
    ]
    for case in cases:
        lines += [f"## {case['id']}", "", str(case["intent"]), ""]
        for record in case["operators"]:
            lines += [f"### {record['operator']}", ""]
            if record["status"] == "rejected":
                lines += [f"Rejected: {record['reason']}", ""]
                continue
            lines += [
                "| # | Track | Objective | Effective anchor percentiles | Branch |",
                "|---:|---|---:|---|---:|",
            ]
            for row in record["top30"]:
                evidence = ", ".join(
                    f"{float(value):.5f}"
                    for value in row["effective_anchor_percentiles"]
                )
                branch = row.get("winning_anchor_index", "")
                lines.append(
                    f"| {row['rank']} | {track_label(row)} | "
                    f"{float(row['objective_score']):.7f} | {evidence} | {branch} |"
                )
            lines.append("")

        for record in case["diagnostic_aggregators"]:
            if record["status"] != "ranked":
                continue
            lines += [
                f"### diagnostic / {record['aggregator']}",
                "",
                "| # | Track | Diagnostic score | Effective anchor percentiles |",
                "|---:|---|---:|---|",
            ]
            for row in record["top30"]:
                evidence = ", ".join(
                    f"{float(value):.5f}"
                    for value in row["effective_anchor_percentiles"]
                )
                lines.append(
                    f"| {row['rank']} | {track_label(row)} | "
                    f"{float(row['objective_score']):.7f} | {evidence} |"
                )
            lines.append("")

        for record in case["direction_diagnostics"]:
            if record["status"] != "ranked":
                continue
            lines += [
                f"### diagnostic / {record['operator']}",
                "",
                "| # | Track | Direction cosine | Effective anchor percentiles |",
                "|---:|---|---:|---|",
            ]
            for row in record["top30"]:
                evidence = ", ".join(
                    f"{float(value):.5f}"
                    for value in row["effective_anchor_percentiles"]
                )
                lines.append(
                    f"| {row['rank']} | {track_label(row)} | "
                    f"{float(row['objective_score']):.7f} | {evidence} |"
                )
            lines.append("")

    if controls:
        lines += ["## Frozen Compound Controls", ""]
        for query in sorted(controls):
            lines += [f"### {query}", "", "| # | Track | Cosine |", "|---:|---|---:|"]
            for row in controls[query]["top30"]:
                lines.append(
                    f"| {row['rank']} | {track_label(row)} | "
                    f"{float(row['objective_score']):.7f} |"
                )
            lines.append("")

    lines += ["## Refine", ""]
    for record in refinements:
        for width in record["widths"]:
            relation = record["secondary"]["relation"]
            percent = float(width["primary_fraction"]) * 100
            lines += [
                f"### {record['id']} / {relation} / {percent:g}%",
                "",
                "| # | Track | Effective B percentile | A percentile |",
                "|---:|---|---:|---:|",
            ]
            for row in width["top30"]:
                values = row["effective_anchor_percentiles"]
                lines.append(
                    f"| {row['rank']} | {track_label(row)} | "
                    f"{float(values[1]):.7f} | {float(values[0]):.7f} |"
                )
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--active-catalog", type=Path, default=DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--phone-report", type=Path, default=DEFAULT_PHONE_REPORT)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    phase_started = total_started
    catalog = parse_active_catalog(args.active_catalog)
    library, db_hash = load_active_library(
        args.db, catalog, verify_hash=not args.skip_db_hash
    )
    load_seconds = time.perf_counter() - phase_started

    text_embeddings, text_hashes, phone_report = load_phone_text_embeddings(
        args.phone_report
    )
    cohort_report = json.loads(args.cohort_report.read_text(encoding="utf-8"))
    if cohort_report["activeCatalog"]["completeTsvSha256"] != EXPECTED_ACTIVE_CATALOG_SHA256:
        raise ValueError("real-cohort report used a different active domain")
    cohort_seed_ids = [int(value) for value in cohort_report["request"]["seedTrackIds"]]
    required_song_ids = {
        int(reference.split(":", 1)[1])
        for definition in (*CASES, *REFINE_CASES, *WEIGHT_SWEEPS)
        for value in (
            definition.get("anchors", ()),
            (definition.get("primary"),),
            (definition.get("secondary"),),
        )
        for item in value
        for reference in (
            item[0] if isinstance(item, tuple) else item,
        )
        if isinstance(reference, str) and reference.startswith("song:")
    }
    if not required_song_ids <= set(cohort_seed_ids):
        raise ValueError("an experiment song anchor is not in the current real cohort")

    cache = AnchorCache(library, text_embeddings)
    replay_started = time.perf_counter()
    replay = phone_replay_checks(library, cache, phone_report)
    replay_seconds = time.perf_counter() - replay_started

    evaluation_started = time.perf_counter()
    controls_needed = sorted(
        {
            str(query)
            for case in CASES
            for query in case.get("compound_controls", ())
        }
    )
    controls = {
        query: direct_text_control(library, cache, query) for query in controls_needed
    }
    cases: list[dict[str, object]] = []
    for position, definition in enumerate(CASES, start=1):
        anchors = [cache.resolve(spec) for spec in definition["anchors"]]
        record: dict[str, object] = {
            "id": definition["id"],
            "intent": definition["intent"],
            "anchors": anchor_records(anchors),
            "compound_controls": list(definition.get("compound_controls", ())),
            "operators": [
                evaluate_operator(library, cache, anchors, operator)
                for operator in OPERATORS
            ],
            "diagnostic_aggregators": [
                evaluate_diagnostic_aggregator(
                    library, cache, anchors, aggregator
                )
                for aggregator in DIAGNOSTIC_AGGREGATORS
            ],
            "direction_diagnostics": [
                evaluate_operator(
                    library, cache, anchors, CALIBRATED_DIRECTION
                )
            ],
        }
        if "missing_exact_compound" in definition:
            record["missing_exact_compound"] = definition["missing_exact_compound"]
        record["compound_embedding_geometry"] = compound_embedding_geometry(
            cache, anchors, record["compound_controls"]
        )
        attach_case_overlaps(record, controls)
        attach_diagnostic_overlaps(record)
        record["pareto"] = pareto_diagnostics(
            library, cache, anchors, record["operators"]
        )
        cases.append(record)
        print(f"case {position}/{len(CASES)} {definition['id']}", flush=True)

    refinements = [
        refine_record(library, cache, definition) for definition in REFINE_CASES
    ]
    print(f"Refine: {len(refinements)} cases x {len(REFINE_WIDTHS)} widths", flush=True)
    weight_sweeps = [
        weight_sweep_record(library, cache, definition) for definition in WEIGHT_SWEEPS
    ]
    print(f"weight sweeps: {len(weight_sweeps)} cases", flush=True)

    endpoint = endpoint_checks(library, cache)
    integrity = validate_rankings(library, cases, refinements)
    evaluation_seconds = time.perf_counter() - evaluation_started

    definitions = {
        "cases": CASES,
        "refine_cases": REFINE_CASES,
        "refine_widths": REFINE_WIDTHS,
        "weight_sweeps": WEIGHT_SWEEPS,
        "operators": OPERATORS,
        "diagnostic_aggregators": DIAGNOSTIC_AGGREGATORS,
        "rrf_k": RRF_K,
        "pareto_3d_max_fronts": PARETO_3D_MAX_FRONTS,
        "calibrated_direction": CALIBRATED_DIRECTION,
        "top_k": TOP_K,
        "diagnostic_k": DIAGNOSTIC_K,
    }
    rankings: dict[str, object] = {
        "schema": "active-composition-eval-v1",
        "scope": {
            "relevance": "embedding-only active-domain retrieval composition",
            "metadata": "labels and diagnostics only",
            "playback_ordering": "not evaluated",
            "queue_mutation": "not performed",
        },
        "inputs": {
            "database_sha256": db_hash,
            "database_embedding_rows": queue_eval.EXPECTED_TRACKS,
            "active_catalog_sha256": sha256_file(args.active_catalog),
            "active_tracks": library.count,
            "quarantined_tracks": int(catalog.quarantined_track_ids.size),
            "database_generation": catalog.database_generation,
            "provider_generation": catalog.provider_generation,
            "phone_report_sha256": sha256_file(args.phone_report),
            "real_cohort_report_sha256": sha256_file(args.cohort_report),
            "phone_text_embedding_sha256": text_hashes,
            "real_cohort_seed_track_ids": cohort_seed_ids,
        },
        "host_identity_semantics": {
            "active_domain_projection": "exact TSV membership before every percentile/objective",
            "equal_cosine_semantics": "production R3 tied upper-CDF percentiles",
            "visible_result_reduction": "all host rows treated as legacy-distinct",
            "ranking_tie_break": "ascending track ID",
            "gap": (
                "The frozen host DB and active TSV do not contain stable-track-span receipt "
                "tokens or production stable ranking tie keys, so verified-copy representative "
                "collapse cannot be reconstructed. The 28-query phone replay nevertheless "
                "matches all 840 ordered top-30 positions."
            ),
        },
        "algorithm_contracts": {
            "all_of": "production weighted geometric mean of signed effective percentiles",
            "either": "production weighted-prefix interleave of positive anchor branches",
            "direction": "cosine to normalized signed weighted centroid",
            "strict_all": "maximum minimum effective percentile; weighted GeoMean then track ID break ties",
            "refine": "B-order inside A's explicit active-domain percentile neighborhood",
            "diagnostic_arithmetic": "weighted arithmetic mean of effective percentiles",
            "diagnostic_harmonic": "weighted harmonic mean (power mean p=-1)",
            "diagnostic_rrf": "weighted reciprocal rank fusion with k=60",
            "diagnostic_pareto": "front/depth structure only; no total result order",
            "diagnostic_calibrated_direction": (
                "normalized signed centroid after inverse active-domain cosine-std scaling"
            ),
        },
        "definitions_sha256": sha256_json(definitions),
        "self_checks": {
            "phone_text_replay": replay,
            "endpoint_and_scale": endpoint,
            "ranking_integrity": integrity,
            "phone_repeat_embeddings_exact": len(text_embeddings),
        },
        "compound_controls": controls,
        "cases": cases,
        "refinements": refinements,
        "weight_sweeps": weight_sweeps,
        "missing_frozen_text_embeddings": ["slow psychedelic guitar"],
    }
    rankings["deterministic_payload_sha256"] = sha256_json(rankings)

    args.output.mkdir(parents=True, exist_ok=True)
    rankings_path = args.output / "rankings.json"
    packet_path = args.output / "listening-packet.md"
    atomic_json(rankings_path, rankings)
    write_listening_packet(packet_path, cases, controls, refinements)

    total_seconds = time.perf_counter() - total_started
    manifest = {
        "schema": "active-composition-run-manifest-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runtime_seconds": {
            "load_and_active_projection": load_seconds,
            "phone_replay": replay_seconds,
            "composition_and_checks": evaluation_seconds,
            "total": total_seconds,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "artifacts": {
            "rankings.json": sha256_file(rankings_path),
            "listening-packet.md": sha256_file(packet_path),
            "evaluator": sha256_file(Path(__file__)),
        },
        "deterministic_payload_sha256": rankings["deterministic_payload_sha256"],
    }
    atomic_json(args.output / "run-manifest.json", manifest)
    print(
        f"complete: {rankings_path} ({total_seconds:.2f}s, "
        f"payload {rankings['deterministic_payload_sha256']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
