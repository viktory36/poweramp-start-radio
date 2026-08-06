#!/usr/bin/env python3
"""Reproduce and test phone queue algorithms against an immutable embedding DB.

This is experiment code, not a second implementation intended for the app. It mirrors
the relevant Kotlin math closely enough to expose behavioral contracts and compare one
change at a time. Metadata is used only for explicit artist/duplicate constraints,
cohort inspection, and output labels. It never enters a relevance score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT / "desktop-indexer" / "audit_raw_data" / "embeddings_phone-latest.db"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "desktop-indexer" / "audit_raw_data" / "v2-discovery" / "queue"
)
EXPECTED_DB_SHA256 = "08dfcec60f7c2e9de4bc6b923d601bd824f80b6251769f6c7bcd8062ce6aa504"
EXPECTED_TRACKS = 80_421
EXPECTED_DIM = 768
DEFAULT_QUEUE_SIZE = 50
DEFAULT_MAX_PER_ARTIST = 8
DEFAULT_MIN_ARTIST_SPACING = 3


@dataclass(frozen=True)
class Library:
    track_ids: np.ndarray
    embeddings: np.ndarray
    artists: tuple[str | None, ...]
    albums: tuple[str | None, ...]
    titles: tuple[str | None, ...]
    durations_ms: np.ndarray
    clusters: np.ndarray
    sources: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    filename_keys: tuple[str, ...]
    file_paths: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.track_ids.size)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_only_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve()}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def load_library(db_path: Path, verify_hash: bool = True) -> tuple[Library, str]:
    actual_path = db_path.resolve()
    if not actual_path.is_file():
        raise FileNotFoundError(actual_path)

    print(f"Verifying immutable database: {actual_path}", file=sys.stderr)
    db_hash = sha256_file(actual_path) if verify_hash else "not-checked"
    if verify_hash and db_hash != EXPECTED_DB_SHA256:
        raise ValueError(
            f"database SHA-256 mismatch: expected {EXPECTED_DB_SHA256}, got {db_hash}"
        )

    with read_only_connection(actual_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM tracks t "
            "INNER JOIN embeddings_clamp3 e ON e.track_id = t.id"
        ).fetchone()[0]
        if verify_hash and count != EXPECTED_TRACKS:
            raise ValueError(f"expected {EXPECTED_TRACKS} joined tracks, found {count}")

        rows = connection.execute(
            "SELECT t.id, t.artist, t.album, t.title, t.duration_ms, t.cluster_id, "
            "COALESCE(t.source, 'desktop') AS source, t.metadata_key, t.filename_key, "
            "t.file_path, e.embedding "
            "FROM tracks t INNER JOIN embeddings_clamp3 e ON e.track_id = t.id "
            "ORDER BY t.id"
        )

        track_ids = np.empty(count, dtype=np.int64)
        embeddings = np.empty((count, EXPECTED_DIM), dtype=np.float32)
        durations = np.empty(count, dtype=np.int64)
        clusters = np.empty(count, dtype=np.int32)
        artists: list[str | None] = []
        albums: list[str | None] = []
        titles: list[str | None] = []
        sources: list[str] = []
        metadata_keys: list[str] = []
        filename_keys: list[str] = []
        file_paths: list[str] = []

        for index, row in enumerate(rows):
            embedding = np.frombuffer(row["embedding"], dtype="<f4")
            if embedding.size != EXPECTED_DIM:
                raise ValueError(
                    f"track {row['id']} has embedding dim {embedding.size}, expected {EXPECTED_DIM}"
                )
            track_ids[index] = row["id"]
            embeddings[index] = embedding
            durations[index] = row["duration_ms"] if row["duration_ms"] is not None else -1
            clusters[index] = row["cluster_id"] if row["cluster_id"] is not None else -1
            artists.append(row["artist"])
            albums.append(row["album"])
            titles.append(row["title"])
            sources.append(row["source"])
            metadata_keys.append(row["metadata_key"])
            filename_keys.append(row["filename_key"])
            file_paths.append(row["file_path"])

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("embedding matrix contains non-finite or non-unit vectors")

    return (
        Library(
            track_ids=track_ids,
            embeddings=embeddings,
            artists=tuple(artists),
            albums=tuple(albums),
            titles=tuple(titles),
            durations_ms=durations,
            clusters=clusters,
            sources=tuple(sources),
            metadata_keys=tuple(metadata_keys),
            filename_keys=tuple(filename_keys),
            file_paths=tuple(file_paths),
        ),
        db_hash,
    )


def stable_hash_rank(track_id: int, salt: str) -> bytes:
    return hashlib.sha256(f"{salt}:{track_id}".encode("ascii")).digest()


def stable_sample(
    library: Library,
    eligible: Iterable[int],
    count: int,
    salt: str,
    unique_metadata: bool = True,
) -> list[int]:
    ranked = sorted(
        (int(index) for index in eligible),
        key=lambda index: stable_hash_rank(int(library.track_ids[index]), salt),
    )
    result: list[int] = []
    seen_metadata: set[str] = set()
    for index in ranked:
        key = library.metadata_keys[index]
        if unique_metadata and key in seen_metadata:
            continue
        result.append(index)
        seen_metadata.add(key)
        if len(result) == count:
            break
    if len(result) != count:
        raise ValueError(f"cohort {salt} requested {count} tracks, found {len(result)}")
    return result


def build_cohort(library: Library, ordinary_count: int) -> dict[str, list[int]]:
    ordinary_eligible = np.flatnonzero(
        (library.durations_ms >= 90_000)
        & (library.durations_ms <= 600_000)
        & np.fromiter((bool(artist) for artist in library.artists), dtype=bool)
    )
    ordinary = stable_sample(
        library,
        ordinary_eligible,
        ordinary_count,
        salt="phone-2026-07-07-ordinary-v1",
    )
    return {"ordinary": ordinary}


def normalized_artist(artist: str | None) -> str | None:
    # Mirrors Kotlin lowercase() deliberately: no trimming, aliasing, or tag cleanup.
    return artist.lower() if artist is not None else None


def can_add_artist(
    candidate_index: int,
    selected: Sequence[int],
    library: Library,
    max_per_artist: int,
    min_spacing: int,
) -> bool:
    artist = normalized_artist(library.artists[candidate_index])
    if artist is None:
        return True
    count = sum(
        normalized_artist(library.artists[index]) == artist for index in selected
    )
    if count >= max_per_artist:
        return False
    if min_spacing > 0:
        return not any(
            normalized_artist(library.artists[index]) == artist
            for index in selected[-min_spacing:]
        )
    return True


def production_post_filter(
    selected: Sequence[int],
    library: Library,
    max_per_artist: int,
    min_spacing: int,
) -> list[int]:
    result: list[int] = []
    for index in selected:
        if can_add_artist(index, result, library, max_per_artist, min_spacing):
            result.append(index)
    return result


def encode_candidate_artists(
    library: Library,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    codes = np.full(candidates.size, -1, dtype=np.int32)
    code_by_artist: dict[str, int] = {}
    for local_index, library_index in enumerate(candidates):
        artist = normalized_artist(library.artists[int(library_index)])
        if artist is None:
            continue
        code = code_by_artist.get(artist)
        if code is None:
            code = len(code_by_artist)
            code_by_artist[artist] = code
        codes[local_index] = code
    return codes, len(code_by_artist)


def apply_artist_constraints(
    eligible: np.ndarray,
    artist_codes: np.ndarray,
    artist_counts: np.ndarray,
    recent_artist_codes: Sequence[int],
    max_per_artist: int,
) -> None:
    known = artist_codes >= 0
    eligible[known] &= artist_counts[artist_codes[known]] < max_per_artist
    recent_known = [code for code in recent_artist_codes if code >= 0]
    if recent_known:
        eligible &= ~np.isin(artist_codes, recent_known)


def retrieve_candidates(
    library: Library,
    seed_index: int,
    pool_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = library.embeddings @ library.embeddings[seed_index]
    similarities[seed_index] = -np.inf
    pool_size = min(pool_size, library.count - 1)
    partial = np.argpartition(similarities, -pool_size)[-pool_size:]
    # Similarities almost never tie, but track ID makes the experimental order explicit.
    order = np.lexsort((library.track_ids[partial], -similarities[partial]))
    candidates = partial[order].astype(np.int64, copy=False)
    return candidates, similarities[candidates]


def select_mmr(
    library: Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    count: int,
    lambda_: float,
    constraint_aware: bool,
    max_per_artist: int,
    min_spacing: int,
) -> tuple[list[int], list[int]]:
    candidate_embeddings = library.embeddings[candidates]
    artist_codes, artist_code_count = encode_candidate_artists(library, candidates)
    artist_counts = np.zeros(artist_code_count, dtype=np.int32)
    recent_artist_codes: list[int] = []
    remaining = np.ones(candidates.size, dtype=bool)
    max_similarity = np.full(candidates.size, -np.inf, dtype=np.float32)
    selected: list[int] = []
    ranks: list[int] = []

    for step in range(count):
        if not remaining.any():
            break
        penalty = 0.0 if step == 0 else max_similarity
        scores = lambda_ * relevance - (1.0 - lambda_) * penalty
        eligible = remaining.copy()
        if constraint_aware:
            apply_artist_constraints(
                eligible,
                artist_codes,
                artist_counts,
                recent_artist_codes,
                max_per_artist,
            )
        if not eligible.any():
            break
        scores = np.where(eligible, scores, -np.inf)
        best = int(np.argmax(scores))
        selected_index = int(candidates[best])
        selected.append(selected_index)
        ranks.append(best + 1)
        remaining[best] = False
        artist_code = int(artist_codes[best])
        if constraint_aware:
            if artist_code >= 0:
                artist_counts[artist_code] += 1
            recent_artist_codes.append(artist_code)
            if len(recent_artist_codes) > min_spacing:
                recent_artist_codes.pop(0)
        similarities = candidate_embeddings @ library.embeddings[selected_index]
        max_similarity = np.maximum(max_similarity, similarities)

    return selected, ranks


def select_dpp(
    library: Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    count: int,
    constraint_aware: bool,
    max_per_artist: int,
    min_spacing: int,
) -> tuple[list[int], list[int]]:
    candidate_embeddings = library.embeddings[candidates]
    artist_codes, artist_code_count = encode_candidate_artists(library, candidates)
    artist_counts = np.zeros(artist_code_count, dtype=np.int32)
    recent_artist_codes: list[int] = []
    quality = relevance.astype(np.float32, copy=True)
    limit = min(count, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    diagonal = quality * quality
    remaining = np.ones(candidates.size, dtype=bool)
    selected: list[int] = []
    ranks: list[int] = []

    for step in range(limit):
        eligible = remaining.copy()
        if constraint_aware:
            apply_artist_constraints(
                eligible,
                artist_codes,
                artist_counts,
                recent_artist_codes,
                max_per_artist,
            )
        gains = np.where(eligible, diagonal, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break

        selected_index = int(candidates[best])
        selected.append(selected_index)
        ranks.append(best + 1)
        remaining[best] = False
        artist_code = int(artist_codes[best])
        if constraint_aware:
            if artist_code >= 0:
                artist_counts[artist_code] += 1
            recent_artist_codes.append(artist_code)
            if len(recent_artist_codes) > min_spacing:
                recent_artist_codes.pop(0)

        sqrt_gain = math.sqrt(best_gain)
        kernel = quality * quality[best] * (
            candidate_embeddings @ candidate_embeddings[best]
        )
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / sqrt_gain
        update_mask = remaining
        factors[update_mask, step] = new_factor[update_mask]
        diagonal[update_mask] -= new_factor[update_mask] ** 2
        np.maximum(diagonal, 0.0, out=diagonal)
        factors[best, step] = sqrt_gain

    return selected, ranks


def excess_duplicates(values: Iterable[str]) -> int:
    seen: set[str] = set()
    excess = 0
    for value in values:
        if value in seen:
            excess += 1
        else:
            seen.add(value)
    return excess


def queue_metrics(
    library: Library,
    seed_index: int,
    selected: Sequence[int],
    candidate_ranks: Sequence[int],
    requested: int,
) -> dict[str, object]:
    if not selected:
        return {
            "count": 0,
            "complete": False,
            "mean_seed_cosine": None,
            "min_seed_cosine": None,
            "mean_pairwise_cosine": None,
        }

    selected_array = np.asarray(selected, dtype=np.int64)
    embeddings = library.embeddings[selected_array]
    seed_similarities = embeddings @ library.embeddings[seed_index]
    if len(selected) > 1:
        pairwise = embeddings @ embeddings.T
        upper = pairwise[np.triu_indices(len(selected), k=1)]
        mean_pairwise = float(np.mean(upper))
        p95_pairwise = float(np.percentile(upper, 95))
    else:
        mean_pairwise = 0.0
        p95_pairwise = 0.0

    artists = [normalized_artist(library.artists[index]) for index in selected]
    known_artists = [artist for artist in artists if artist is not None]
    known_clusters = {
        int(library.clusters[index])
        for index in selected
        if library.clusters[index] >= 0
    }
    return {
        "count": len(selected),
        "complete": len(selected) == requested,
        "mean_seed_cosine": float(np.mean(seed_similarities)),
        "min_seed_cosine": float(np.min(seed_similarities)),
        "p10_seed_cosine": float(np.percentile(seed_similarities, 10)),
        "mean_pairwise_cosine": mean_pairwise,
        "p95_pairwise_cosine": p95_pairwise,
        "unique_known_artists": len(set(known_artists)),
        "known_artist_count": len(known_artists),
        "cluster_spread": len(known_clusters),
        "missing_cluster_count": sum(library.clusters[index] < 0 for index in selected),
        "median_candidate_rank": float(np.median(candidate_ranks)) if candidate_ranks else None,
        "p90_candidate_rank": (
            float(np.percentile(candidate_ranks, 90)) if candidate_ranks else None
        ),
        "metadata_key_duplicate_excess": excess_duplicates(
            library.metadata_keys[index] for index in selected
        ),
        "file_path_duplicate_excess": excess_duplicates(
            library.file_paths[index].lower() for index in selected
        ),
    }


def track_summary(library: Library, index: int) -> dict[str, object]:
    return {
        "index": int(index),
        "track_id": int(library.track_ids[index]),
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "duration_ms": int(library.durations_ms[index]),
        "cluster_id": int(library.clusters[index]),
        "source": library.sources[index],
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=True,
                sort_keys=True,
                default=json_numpy_scalar,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def read_completed_seeds(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(int(json.loads(line)["seed"]["track_id"]))
    return completed


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            default=json_numpy_scalar,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def json_numpy_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def variant_record(
    library: Library,
    seed_index: int,
    selected: Sequence[int],
    ranks: Sequence[int],
    requested: int,
    elapsed_ms: float,
) -> dict[str, object]:
    return {
        "metrics": queue_metrics(library, seed_index, selected, ranks, requested),
        "elapsed_ms": elapsed_ms,
        "track_ids": [int(library.track_ids[index]) for index in selected],
        "candidate_ranks": list(ranks),
    }


def run_constraints(
    library: Library,
    seeds: Sequence[int],
    output_file: Path,
    pool_size: int,
    requested: int,
) -> None:
    completed = read_completed_seeds(output_file)
    variants = (
        ("mmr_postfilter", "mmr", False),
        ("mmr_constraint_aware", "mmr", True),
        ("dpp_postfilter", "dpp", False),
        ("dpp_constraint_aware", "dpp", True),
    )
    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"constraints {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        candidates, relevance = retrieve_candidates(library, seed_index, pool_size)
        record: dict[str, object] = {
            "seed": track_summary(library, seed_index),
            "pool_size": pool_size,
            "requested": requested,
            "variants": {},
        }
        for name, selector, constraint_aware in variants:
            started = time.perf_counter()
            if selector == "mmr":
                selected, ranks = select_mmr(
                    library,
                    candidates,
                    relevance,
                    requested,
                    lambda_=0.4,
                    constraint_aware=constraint_aware,
                    max_per_artist=DEFAULT_MAX_PER_ARTIST,
                    min_spacing=DEFAULT_MIN_ARTIST_SPACING,
                )
            else:
                selected, ranks = select_dpp(
                    library,
                    candidates,
                    relevance,
                    requested,
                    constraint_aware=constraint_aware,
                    max_per_artist=DEFAULT_MAX_PER_ARTIST,
                    min_spacing=DEFAULT_MIN_ARTIST_SPACING,
                )
            if not constraint_aware:
                keep = production_post_filter(
                    selected,
                    library,
                    DEFAULT_MAX_PER_ARTIST,
                    DEFAULT_MIN_ARTIST_SPACING,
                )
                rank_by_index = dict(zip(selected, ranks))
                selected = keep
                ranks = [rank_by_index[index] for index in selected]
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record["variants"][name] = variant_record(
                library, seed_index, selected, ranks, requested, elapsed_ms
            )
        append_jsonl(output_file, record)
        print(
            f"constraints {position}/{len(seeds)} seed={seed_id} saved",
            file=sys.stderr,
        )


def run_lambda(
    library: Library,
    seeds: Sequence[int],
    output_file: Path,
    pool_size: int,
    requested: int,
) -> None:
    completed = read_completed_seeds(output_file)
    lambdas = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"lambda {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        candidates, relevance = retrieve_candidates(library, seed_index, pool_size)
        record: dict[str, object] = {
            "seed": track_summary(library, seed_index),
            "pool_size": pool_size,
            "requested": requested,
            "variants": {},
        }
        for lambda_ in lambdas:
            started = time.perf_counter()
            selected, ranks = select_mmr(
                library,
                candidates,
                relevance,
                requested,
                lambda_=lambda_,
                constraint_aware=True,
                max_per_artist=DEFAULT_MAX_PER_ARTIST,
                min_spacing=DEFAULT_MIN_ARTIST_SPACING,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record["variants"][f"lambda_{lambda_:0.1f}"] = variant_record(
                library, seed_index, selected, ranks, requested, elapsed_ms
            )
        append_jsonl(output_file, record)
        print(f"lambda {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def run_pools(
    library: Library,
    seeds: Sequence[int],
    output_file: Path,
    pool_size: int,
    requested: int,
) -> None:
    completed = read_completed_seeds(output_file)
    pool_sizes = tuple(sorted({100, 250, 500, 1_000, pool_size}))
    for position, seed_index in enumerate(seeds, start=1):
        seed_id = int(library.track_ids[seed_index])
        if seed_id in completed:
            print(f"pools {position}/{len(seeds)} seed={seed_id} resumed", file=sys.stderr)
            continue
        all_candidates, all_relevance = retrieve_candidates(library, seed_index, pool_size)
        record: dict[str, object] = {
            "seed": track_summary(library, seed_index),
            "reference_pool_size": pool_size,
            "requested": requested,
            "variants": {},
        }
        for current_pool in pool_sizes:
            candidates = all_candidates[:current_pool]
            relevance = all_relevance[:current_pool]
            started = time.perf_counter()
            selected, ranks = select_mmr(
                library,
                candidates,
                relevance,
                requested,
                lambda_=0.4,
                constraint_aware=True,
                max_per_artist=DEFAULT_MAX_PER_ARTIST,
                min_spacing=DEFAULT_MIN_ARTIST_SPACING,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            record["variants"][f"pool_{current_pool}"] = variant_record(
                library, seed_index, selected, ranks, requested, elapsed_ms
            )
        append_jsonl(output_file, record)
        print(f"pools {position}/{len(seeds)} seed={seed_id} saved", file=sys.stderr)


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize_jsonl(path: Path) -> dict[str, object]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    variants = sorted(records[0]["variants"]) if records else []
    summary: dict[str, object] = {"records": len(records), "variants": {}}
    for variant in variants:
        metrics = [record["variants"][variant]["metrics"] for record in records]
        elapsed = [float(record["variants"][variant]["elapsed_ms"]) for record in records]
        numeric_keys = (
            "count",
            "mean_seed_cosine",
            "min_seed_cosine",
            "mean_pairwise_cosine",
            "unique_known_artists",
            "cluster_spread",
            "median_candidate_rank",
            "metadata_key_duplicate_excess",
            "file_path_duplicate_excess",
        )
        aggregate: dict[str, object] = {
            "complete_rate": sum(bool(metric["complete"]) for metric in metrics) / len(metrics),
            "elapsed_ms_median": percentile(elapsed, 50),
            "elapsed_ms_p95": percentile(elapsed, 95),
        }
        for key in numeric_keys:
            values = [float(metric[key]) for metric in metrics if metric.get(key) is not None]
            aggregate[f"{key}_mean"] = float(np.mean(values)) if values else None
            aggregate[f"{key}_p50"] = percentile(values, 50)
            aggregate[f"{key}_p05"] = percentile(values, 5)
            aggregate[f"{key}_p95"] = percentile(values, 95)
        summary["variants"][variant] = aggregate
    return summary


def write_manifest(
    output_dir: Path,
    library: Library,
    db_path: Path,
    db_hash: str,
    cohort: dict[str, list[int]],
    args: argparse.Namespace,
) -> None:
    value = {
        "database": str(db_path.resolve()),
        "database_sha256": db_hash,
        "track_count": library.count,
        "embedding_dim": library.dim,
        "cohort": {
            name: [track_summary(library, index) for index in indices]
            for name, indices in cohort.items()
        },
        "parameters": {
            "queue_size": args.queue_size,
            "pool_size": args.pool_size,
            "max_per_artist": DEFAULT_MAX_PER_ARTIST,
            "min_artist_spacing": DEFAULT_MIN_ARTIST_SPACING,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    atomic_json(output_dir / "manifest.json", value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    parser.add_argument("--pool-size", type=int, default=0, help="0 mirrors the app's 2%% rule")
    parser.add_argument(
        "--experiments",
        default="constraints,lambda,pools",
        help="comma-separated: constraints,lambda,pools",
    )
    parser.add_argument("--skip-hash", action="store_true", help="development only")
    parser.add_argument("--force", action="store_true", help="discard checkpoints first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seeds <= 0 or args.queue_size <= 0:
        raise ValueError("--seeds and --queue-size must be positive")
    library, db_hash = load_library(args.db, verify_hash=not args.skip_hash)
    pool_size = args.pool_size or max(100, int(library.count * 0.02))
    if pool_size < args.queue_size:
        raise ValueError("candidate pool must be at least as large as the queue")
    args.pool_size = pool_size

    cohort = build_cohort(library, args.seeds)
    seeds = cohort["ordinary"]
    args.output.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output, library, args.db, db_hash, cohort, args)

    requested_experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    runners = {
        "constraints": run_constraints,
        "lambda": run_lambda,
        "pools": run_pools,
    }
    unknown = set(requested_experiments) - set(runners)
    if unknown:
        raise ValueError(f"unknown experiments: {', '.join(sorted(unknown))}")

    for name in requested_experiments:
        output_file = args.output / f"{name}.jsonl"
        if args.force and output_file.exists():
            output_file.unlink()
        print(f"Running {name} -> {output_file}", file=sys.stderr)
        runners[name](library, seeds, output_file, pool_size, args.queue_size)
        atomic_json(args.output / f"{name}-summary.json", summarize_jsonl(output_file))

    print(f"Complete. Results: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
