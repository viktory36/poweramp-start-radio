#!/usr/bin/env python3
"""Compare fixed-window CLaMP3 policies with full cached MERT sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

import v2_queue_eval as queue_eval


REPO_ROOT = queue_eval.REPO_ROOT
DEFAULT_FEATURES = (
    REPO_ROOT / "desktop-indexer" / "audit_raw_data" / "clamp3_cache" / "mert_features"
)
DEFAULT_FULL_EMBEDDINGS = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "clamp3_cache"
    / "clamp3_audio_embs.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "desktop-indexer" / "audit_raw_data" / "v2-discovery" / "sampling"
)
POOL_SIZE = 1_608
QUEUE_SIZE = 50
MMR_LAMBDA = 0.4
POLICIES: tuple[tuple[str, str, int | None], ...] = (
    ("full", "full", None),
    ("prefix_2", "prefix", 2),
    ("prefix_6", "prefix", 6),
    ("prefix_24", "prefix", 24),
    ("uniform_6", "uniform", 6),
    ("uniform_12", "uniform", 12),
    ("uniform_24", "uniform", 24),
    ("uniform_48", "uniform", 48),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_manifest_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def uniform_indices(total: int, requested: int) -> np.ndarray:
    if total <= 0 or requested <= 0:
        return np.empty(0, dtype=np.int64)
    if requested >= total:
        return np.arange(total, dtype=np.int64)
    raw = np.linspace(0, total - 1, requested)
    indices = np.rint(raw).astype(np.int64)
    # For requested <= total, rounded linspace should be unique. Keep the contract
    # explicit in case NumPy behavior changes.
    indices = np.unique(indices)
    if indices.size != requested:
        available = [index for index in range(total) if index not in set(indices.tolist())]
        indices = np.sort(
            np.concatenate(
                (indices, np.asarray(available[: requested - indices.size], dtype=np.int64))
            )
        )
    return indices


def policy_indices(total: int, kind: str, requested: int | None) -> np.ndarray:
    if kind == "full":
        return np.arange(total, dtype=np.int64)
    if requested is None:
        raise ValueError("requested windows missing")
    if kind == "prefix":
        return np.arange(min(total, requested), dtype=np.int64)
    if kind == "uniform":
        return uniform_indices(total, requested)
    raise ValueError(kind)


def windows_by_cache_key(feature_dir: Path, full_keys: Sequence[str]) -> dict[str, Path]:
    key_by_stem: dict[str, list[str]] = {}
    for key in full_keys:
        key_by_stem.setdefault(Path(key).stem.casefold(), []).append(key)
    result: dict[str, Path] = {}
    for path in sorted(feature_dir.glob("*.npy")):
        matches = key_by_stem.get(path.stem.casefold(), [])
        if len(matches) == 1:
            result[matches[0]] = path
    return result


def windows_path_to_phone_index(library: queue_eval.Library) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for index, file_path in enumerate(library.file_paths):
        basename = file_path.replace("\\", "/").rsplit("/", 1)[-1].casefold()
        result.setdefault(basename, []).append(index)
    return result


def top_indices(
    library: queue_eval.Library,
    query: np.ndarray,
    exclude_index: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = library.embeddings @ query
    similarities[exclude_index] = -np.inf
    count = min(count, library.count - 1)
    partial = np.argpartition(similarities, -count)[-count:]
    order = np.lexsort((library.track_ids[partial], -similarities[partial]))
    chosen = partial[order].astype(np.int64, copy=False)
    return chosen, similarities


def rank_of_indices(
    similarities: np.ndarray,
    library: queue_eval.Library,
    target_indices: Sequence[int],
) -> list[int]:
    ranks: list[int] = []
    for index in target_indices:
        score = similarities[index]
        better = int(np.count_nonzero(similarities > score))
        tied_before = int(
            np.count_nonzero(
                (similarities == score) & (library.track_ids < library.track_ids[index])
            )
        )
        ranks.append(better + tied_before + 1)
    return ranks


def overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return len(set(int(value) for value in left) & set(int(value) for value in right))


def retrieve_for_query(
    library: queue_eval.Library, query: np.ndarray, exclude_index: int
) -> tuple[np.ndarray, np.ndarray]:
    return top_indices(library, query, exclude_index, POOL_SIZE)


def mmr_for_query(
    library: queue_eval.Library, query: np.ndarray, exclude_index: int
) -> list[int]:
    candidates, all_similarities = retrieve_for_query(library, query, exclude_index)
    relevance = all_similarities[candidates]
    selected, _ = queue_eval.select_mmr(
        library,
        candidates,
        relevance,
        QUEUE_SIZE,
        lambda_=MMR_LAMBDA,
        constraint_aware=True,
        max_per_artist=queue_eval.DEFAULT_MAX_PER_ARTIST,
        min_spacing=queue_eval.DEFAULT_MIN_ARTIST_SPACING,
    )
    return selected


def append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"bad checkpoint line {line_number}: {error}") from error
            result[str(row["cache_key"])] = row
    return result


def encode_policy(generator: object, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    selected = np.asarray(features[:, indices, :], dtype=np.float32)
    embedding = generator.encode_mert_features(selected)
    if embedding is None:
        raise RuntimeError("CLaMP3 audio encoding returned null")
    result = np.asarray(embedding, dtype=np.float32)
    if result.shape != (queue_eval.EXPECTED_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid output embedding: {result.shape}")
    return result


def evaluate_track(
    generator: object,
    library: queue_eval.Library,
    cache_key: str,
    feature_path: Path,
    phone_index: int,
    cached_full_embedding: np.ndarray,
) -> dict[str, object]:
    features = np.load(feature_path)
    if features.ndim != 3 or features.shape[0] != 1 or features.shape[2] != library.dim:
        raise ValueError(f"unexpected feature shape {features.shape}: {feature_path}")
    total_windows = int(features.shape[1])
    phone_embedding = library.embeddings[phone_index]

    generated: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    indices_by_policy: dict[str, np.ndarray] = {}
    for name, kind, requested in POLICIES:
        indices = policy_indices(total_windows, kind, requested)
        started = time.perf_counter()
        generated[name] = encode_policy(generator, features, indices)
        timings[name] = (time.perf_counter() - started) * 1000.0
        indices_by_policy[name] = indices

    full_embedding = generated["full"]
    full_top50, full_similarities = top_indices(library, full_embedding, phone_index, 50)
    phone_top50, _ = top_indices(library, phone_embedding, phone_index, 50)
    full_mmr = mmr_for_query(library, full_embedding, phone_index)

    variants: dict[str, object] = {}
    for name, _, _ in POLICIES:
        embedding = generated[name]
        top50, similarities = top_indices(library, embedding, phone_index, 50)
        sampled_mmr = full_mmr if name == "full" else mmr_for_query(
            library, embedding, phone_index
        )
        full_top10_ranks = rank_of_indices(similarities, library, full_top50[:10])
        variants[name] = {
            "window_indices": indices_by_policy[name].tolist(),
            "window_count": int(indices_by_policy[name].size),
            "window_fraction": float(indices_by_policy[name].size / total_windows),
            "encode_ms": timings[name],
            "cosine_to_reencoded_full": float(embedding @ full_embedding),
            "cosine_to_cached_full": float(embedding @ cached_full_embedding),
            "cosine_to_phone": float(embedding @ phone_embedding),
            "top10_overlap_with_reencoded_full": overlap(top50[:10], full_top50[:10]),
            "top50_overlap_with_reencoded_full": overlap(top50, full_top50),
            "top50_overlap_with_phone": overlap(top50, phone_top50),
            "full_top10_median_rank": float(np.median(full_top10_ranks)),
            "full_top10_max_rank": max(full_top10_ranks),
            "mmr_overlap_with_reencoded_full": overlap(sampled_mmr, full_mmr),
            "mmr_jaccard_with_reencoded_full": (
                overlap(sampled_mmr, full_mmr)
                / len(set(sampled_mmr) | set(full_mmr))
            ),
            "top50_track_ids": [int(library.track_ids[index]) for index in top50],
            "mmr_track_ids": [int(library.track_ids[index]) for index in sampled_mmr],
            "embedding": embedding.tolist(),
        }

    return {
        "cache_key": cache_key,
        "feature_file": feature_path.name,
        "total_windows": total_windows,
        "duration_seconds_from_windows": total_windows * 5,
        "phone_track": queue_eval.track_summary(library, phone_index),
        "cached_full_cosine_to_phone": float(cached_full_embedding @ phone_embedding),
        "reencoded_full_cosine_to_cached_full": float(full_embedding @ cached_full_embedding),
        "reencoded_full_cosine_to_phone": float(full_embedding @ phone_embedding),
        "reencoded_full_top50_overlap_with_phone": overlap(full_top50, phone_top50),
        "variants": variants,
    }


def percentile_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    metrics = (
        "window_fraction",
        "encode_ms",
        "cosine_to_reencoded_full",
        "cosine_to_phone",
        "top10_overlap_with_reencoded_full",
        "top50_overlap_with_reencoded_full",
        "top50_overlap_with_phone",
        "full_top10_median_rank",
        "full_top10_max_rank",
        "mmr_overlap_with_reencoded_full",
        "mmr_jaccard_with_reencoded_full",
    )
    slices: dict[str, object] = {}
    for minimum_windows in (1, 24, 48, 72, 120):
        eligible = [row for row in records if int(row["total_windows"]) >= minimum_windows]
        if not eligible:
            continue
        policy_rows: dict[str, object] = {}
        for name, _, _ in POLICIES:
            policy_rows[name] = {
                metric: percentile_summary(
                    [float(row["variants"][name][metric]) for row in eligible]
                )
                for metric in metrics
            }
        slices[f"min_{minimum_windows}_windows"] = {
            "count": len(eligible),
            "policies": policy_rows,
        }
    return {
        "slices": slices,
        "full_validation": {
            "reencoded_to_cached": percentile_summary(
                [float(row["reencoded_full_cosine_to_cached_full"]) for row in records]
            ),
            "reencoded_to_phone": percentile_summary(
                [float(row["reencoded_full_cosine_to_phone"]) for row in records]
            ),
            "top50_overlap_phone": percentile_summary(
                [float(row["reencoded_full_top50_overlap_with_phone"]) for row in records]
            ),
        },
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_worst_cases(
    path: Path, library: queue_eval.Library, records: Sequence[dict[str, object]]
) -> None:
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    lines = ["# Sampling Worst Cases", ""]
    for policy, _, _ in POLICIES:
        if policy == "full":
            continue
        ordered = sorted(
            records,
            key=lambda row: (
                int(row["variants"][policy]["top50_overlap_with_reencoded_full"]),
                int(row["phone_track"]["track_id"]),
            ),
        )[:5]
        lines.extend((f"## {policy}", ""))
        for row in ordered:
            variant = row["variants"][policy]
            track = row["phone_track"]
            lines.append(
                f"### {track['artist'] or '?'} - {track['title'] or '?'} "
                f"({row['total_windows']} windows)"
            )
            lines.append("")
            lines.append(
                f"cos(full)={variant['cosine_to_reencoded_full']:.6f}; "
                f"top50 overlap={variant['top50_overlap_with_reencoded_full']}/50; "
                f"MMR overlap={variant['mmr_overlap_with_reencoded_full']}/50"
            )
            lines.append("")
            lines.append("Sampled top 10:")
            for rank, track_id in enumerate(variant["top50_track_ids"][:10], start=1):
                index = by_id[int(track_id)]
                lines.append(
                    f"{rank}. {library.artists[index] or '?'} - {library.titles[index] or '?'}"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--full-embeddings", type=Path, default=DEFAULT_FULL_EMBEDDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="development smoke run")
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library, db_hash = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    full_archive = np.load(args.full_embeddings)
    paths_by_key = windows_by_cache_key(args.features, full_archive.files)
    indices_by_basename = windows_path_to_phone_index(library)
    cohort: list[tuple[str, Path, int]] = []
    for cache_key, feature_path in paths_by_key.items():
        matches = indices_by_basename.get(cache_key.casefold(), [])
        if len(matches) == 1:
            cohort.append((cache_key, feature_path, matches[0]))
    cohort.sort(key=lambda item: int(library.track_ids[item[2]]))
    if args.limit > 0:
        cohort = cohort[: args.limit]
    if not cohort:
        raise ValueError("no uniquely mapped cache tracks")

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "track-results.jsonl"
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
    completed = load_checkpoint(checkpoint_path)

    from huggingface_hub import hf_hub_download
    import torch
    from poweramp_indexer.embeddings_clamp3 import (
        CLAMP3_WEIGHTS_FILENAME,
        CLaMP3EmbeddingGenerator,
    )

    model_path = Path(
        hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    ).resolve()
    generator = CLaMP3EmbeddingGenerator(fp16=False)
    try:
        for position, (cache_key, feature_path, phone_index) in enumerate(cohort, start=1):
            if cache_key in completed:
                print(f"{position}/{len(cohort)} {cache_key}: resumed", flush=True)
                continue
            row = evaluate_track(
                generator,
                library,
                cache_key,
                feature_path,
                phone_index,
                np.asarray(full_archive[cache_key], dtype=np.float32),
            )
            append_checkpoint(checkpoint_path, row)
            completed[cache_key] = row
            prefix24 = row["variants"]["prefix_24"]
            uniform24 = row["variants"]["uniform_24"]
            print(
                f"{position}/{len(cohort)} {cache_key}: windows={row['total_windows']} "
                f"prefix24={prefix24['top50_overlap_with_reencoded_full']}/50 "
                f"uniform24={uniform24['top50_overlap_with_reencoded_full']}/50",
                flush=True,
            )
    finally:
        generator.unload_models()

    records = [completed[cache_key] for cache_key, _, _ in cohort]
    summary = summarize_records(records)
    manifest = {
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "tracks": library.count,
            "dim": library.dim,
        },
        "features": {
            "path": str(args.features.resolve()),
            "matched_count": len(cohort),
            "matched_manifest_sha256": directory_manifest_hash(
                [feature_path for _, feature_path, _ in cohort]
            ),
        },
        "cached_full_embeddings": {
            "path": str(args.full_embeddings.resolve()),
            "sha256": sha256_file(args.full_embeddings),
        },
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
        },
        "policies": [
            {"name": name, "kind": kind, "requested_windows": requested}
            for name, kind, requested in POLICIES
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "pool_size": POOL_SIZE,
        "queue_size": QUEUE_SIZE,
        "mmr_lambda": MMR_LAMBDA,
        "summary": summary,
    }
    atomic_json(args.output / "summary.json", manifest)
    write_worst_cases(args.output / "worst-cases.md", library, records)
    print(f"Complete. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
