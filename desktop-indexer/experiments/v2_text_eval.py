#!/usr/bin/env python3
"""Frozen CLaMP3 text-retrieval audit against the immutable phone database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

import v2_queue_eval as queue_eval
from poweramp_indexer.embeddings_clamp3 import (
    CLAMP3_WEIGHTS_FILENAME,
    CLaMP3EmbeddingGenerator,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = SCRIPT_DIR / "v2_text_prompts.json"
DEFAULT_OUTPUT = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_rows(spec: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in spec["described_tracks"]:
        rows.append(
            {
                "id": f"described/{item['id']}",
                "kind": "described_track",
                "prompt": item["prompt"],
                "target_track_ids": item["target_track_ids"],
            }
        )
    for item in spec["open_queries"]:
        rows.append(
            {
                "id": f"open/{item['id']}",
                "kind": "open_query",
                "prompt": item["prompt"],
            }
        )
    for group in spec["language_groups"]:
        for language, prompt in group["prompts"].items():
            rows.append(
                {
                    "id": f"language/{group['id']}/{language}",
                    "kind": "language",
                    "language_group": group["id"],
                    "language": language,
                    "prompt": prompt,
                }
            )
    for index, prompt in enumerate(spec["operator_anchors"]):
        rows.append(
            {
                "id": f"anchor/{index:02d}",
                "kind": "operator_anchor",
                "prompt": prompt,
            }
        )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prompt IDs are not unique")
    return rows


def load_checkpoint(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid checkpoint line {line_number}: {error}") from error
            records[str(row["id"])] = row
    return records


def append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def rank_order(
    similarities: np.ndarray,
    track_ids: np.ndarray,
) -> np.ndarray:
    return np.lexsort((track_ids, -similarities))


def top_results(
    library: queue_eval.Library,
    similarities: np.ndarray,
    count: int,
) -> list[dict[str, object]]:
    order = rank_order(similarities, library.track_ids)[:count]
    return [
        {
            "rank": rank,
            "track_id": int(library.track_ids[index]),
            "score": float(similarities[index]),
            "artist": library.artists[index],
            "album": library.albums[index],
            "title": library.titles[index],
            "duration_ms": int(library.durations_ms[index]),
            "source": library.sources[index],
            "metadata_key": library.metadata_keys[index],
            "file_path": library.file_paths[index],
        }
        for rank, index in enumerate(order, start=1)
    ]


def exact_target_rank(
    library: queue_eval.Library,
    similarities: np.ndarray,
    target_track_ids: Iterable[int],
) -> dict[str, object]:
    index_by_id = {
        int(track_id): index for index, track_id in enumerate(library.track_ids)
    }
    target_indices = [index_by_id[int(track_id)] for track_id in target_track_ids]
    order = rank_order(similarities, library.track_ids)
    ranks = np.empty(library.count, dtype=np.int64)
    ranks[order] = np.arange(1, library.count + 1)
    best = min(target_indices, key=lambda index: int(ranks[index]))
    return {
        "best_rank": int(ranks[best]),
        "best_track_id": int(library.track_ids[best]),
        "best_score": float(similarities[best]),
        "all_target_ranks": {
            str(int(library.track_ids[index])): int(ranks[index])
            for index in target_indices
        },
    }


def duplicate_metrics(results: list[dict[str, object]], count: int = 20) -> dict[str, int]:
    prefix = results[:count]
    metadata_keys = [str(row["metadata_key"]) for row in prefix]
    paths = [str(row["file_path"]).lower() for row in prefix]
    return {
        "top_count": count,
        "metadata_key_duplicate_excess": len(metadata_keys) - len(set(metadata_keys)),
        "file_path_duplicate_excess": len(paths) - len(set(paths)),
    }


def score_distribution(similarities: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(similarities, [0, 1, 10, 50, 90, 99, 99.9, 100])
    return {
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "p10": float(percentiles[2]),
        "p50": float(percentiles[3]),
        "p90": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "p999": float(percentiles[6]),
        "max": float(percentiles[7]),
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
    }


def evaluate_prompt(
    generator: CLaMP3EmbeddingGenerator,
    library: queue_eval.Library,
    prompt_row: dict[str, object],
    top_k: int,
) -> dict[str, object]:
    started = time.perf_counter()
    embedding_list = generator.embed_text(str(prompt_row["prompt"]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if embedding_list is None:
        raise RuntimeError(f"text embedding failed: {prompt_row['id']}")
    embedding = np.asarray(embedding_list, dtype=np.float32)
    similarities = library.embeddings @ embedding
    results = top_results(library, similarities, top_k)
    record = dict(prompt_row)
    record.update(
        {
            "embedding": embedding.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "inference_ms": elapsed_ms,
            "score_distribution": score_distribution(similarities),
            "duplicate_metrics": duplicate_metrics(results),
            "top_results": results,
        }
    )
    target_ids = prompt_row.get("target_track_ids")
    if target_ids:
        record["target"] = exact_target_rank(library, similarities, target_ids)
    return record


def set_overlap(a: list[int], b: list[int]) -> dict[str, float | int]:
    left = set(a)
    right = set(b)
    intersection = len(left & right)
    union = len(left | right)
    return {
        "intersection": intersection,
        "jaccard": intersection / union if union else 1.0,
    }


def language_summary(
    records: dict[str, dict[str, object]],
    spec: dict[str, object],
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for group in spec["language_groups"]:
        languages = list(group["prompts"])
        for left_position, left_language in enumerate(languages):
            for right_language in languages[left_position + 1 :]:
                left = records[f"language/{group['id']}/{left_language}"]
                right = records[f"language/{group['id']}/{right_language}"]
                left_embedding = np.asarray(left["embedding"], dtype=np.float32)
                right_embedding = np.asarray(right["embedding"], dtype=np.float32)
                row: dict[str, object] = {
                    "group": group["id"],
                    "left": left_language,
                    "right": right_language,
                    "embedding_cosine": float(left_embedding @ right_embedding),
                }
                for depth in (10, 20, 50):
                    left_ids = [
                        int(item["track_id"]) for item in left["top_results"][:depth]
                    ]
                    right_ids = [
                        int(item["track_id"]) for item in right["top_results"][:depth]
                    ]
                    row[f"top_{depth}"] = set_overlap(left_ids, right_ids)
                summary.append(row)
    return summary


def described_summary(records: dict[str, dict[str, object]]) -> dict[str, object]:
    rows = [row for row in records.values() if row["kind"] == "described_track"]
    ranks = [int(row["target"]["best_rank"]) for row in rows]
    return {
        "count": len(rows),
        "median_best_target_rank": float(np.median(ranks)),
        "geometric_mean_best_target_rank": float(np.exp(np.mean(np.log(ranks)))),
        "within_top_10": sum(rank <= 10 for rank in ranks),
        "within_top_50": sum(rank <= 50 for rank in ranks),
        "within_top_100": sum(rank <= 100 for rank in ranks),
        "within_top_1000": sum(rank <= 1000 for rank in ranks),
        "per_prompt": [
            {
                "id": row["id"],
                "prompt": row["prompt"],
                **row["target"],
            }
            for row in sorted(rows, key=lambda value: int(value["target"]["best_rank"]))
        ],
    }


def resolve_checkpoint() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    parser.add_argument("--skip-model-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 50:
        raise ValueError("--top-k must be at least 50 for the frozen analysis")
    spec = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = prompt_rows(spec)
    library, db_hash = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    model_path = resolve_checkpoint()
    model_hash = "not-checked" if args.skip_model_hash else sha256_file(model_path)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output / "prompt-results.jsonl"
    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()
    completed = load_checkpoint(checkpoint_path)

    generator = CLaMP3EmbeddingGenerator(fp16=False)
    try:
        for position, prompt in enumerate(prompts, start=1):
            prompt_id = str(prompt["id"])
            if prompt_id in completed:
                print(f"{position}/{len(prompts)} {prompt_id}: resumed", file=sys.stderr)
                continue
            record = evaluate_prompt(generator, library, prompt, args.top_k)
            append_checkpoint(checkpoint_path, record)
            completed[prompt_id] = record
            print(
                f"{position}/{len(prompts)} {prompt_id}: "
                f"{record['inference_ms']:.1f} ms",
                file=sys.stderr,
            )
    finally:
        generator.unload_models()

    ordered_records = {str(prompt["id"]): completed[str(prompt["id"])] for prompt in prompts}
    summary = {
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "track_count": library.count,
            "embedding_dim": library.dim,
        },
        "model": {
            "name": "sander-wood/clamp3",
            "checkpoint": str(model_path),
            "checkpoint_sha256": model_hash,
            "text_embedding_dim": 768,
        },
        "prompt_spec": {
            "path": str(args.prompts.resolve()),
            "sha256": sha256_file(args.prompts),
            "version": spec["version"],
            "frozen_before_ranking": spec["frozen_before_ranking"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "prompt_count": len(prompts),
        "described_tracks": described_summary(ordered_records),
        "language_pairs": language_summary(ordered_records, spec),
        "inference_ms": {
            "median": float(
                np.median([float(row["inference_ms"]) for row in ordered_records.values()])
            ),
            "p95": float(
                np.percentile(
                    [float(row["inference_ms"]) for row in ordered_records.values()], 95
                )
            ),
        },
    }
    atomic_json(args.output / "summary.json", summary)
    print(f"Complete. Results: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
