#!/usr/bin/env python3
"""Post-preregistered text retrieval diagnostics against the immutable phone DB.

This script never changes the database. It verifies the frozen T01-T05 evidence,
replays every original ranking from its stored text embedding, and runs a separately
declared follow-up prompt-sensitivity set.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import platform
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np

import v2_queue_eval as queue_eval
import v2_text_eval as initial_eval
from poweramp_indexer.embeddings_clamp3 import CLaMP3EmbeddingGenerator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = SCRIPT_DIR / "v2_text_followup_prompts.json"
DEFAULT_INITIAL_OUTPUT = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text"
)
DEFAULT_OUTPUT = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text-followup"
)
EXPECTED_INITIAL_RESULTS_SHA256 = (
    "f57a2dc02b73b7885830fb1298e98c31ec6c3d8289c2399c7bb7f2d6c75d3696"
)
EXPECTED_INITIAL_SUMMARY_SHA256 = (
    "9b362267cf724f33e469f883a2647f1fde3fbfdab9344119be811a3d5c96206e"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number}: {error}") from error
    return rows


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonical_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def result_indices(
    library: queue_eval.Library,
    similarities: np.ndarray,
    count: int,
) -> np.ndarray:
    return initial_eval.rank_order(similarities, library.track_ids)[:count]


def overlap(left: Iterable[int], right: Iterable[int]) -> dict[str, float | int]:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return {
        "intersection": intersection,
        "overlap_fraction": intersection / len(left_set) if left_set else 1.0,
        "jaccard": intersection / union if union else 1.0,
    }


def crowding_metrics(
    library: queue_eval.Library,
    ordered_indices: np.ndarray,
    depth: int,
) -> dict[str, object]:
    indices = [int(value) for value in ordered_indices[:depth]]
    metadata = [library.metadata_keys[index] for index in indices]
    paths = [library.file_paths[index].casefold() for index in indices]
    identities: list[str] = []
    known_artists: list[str] = []
    for index in indices:
        artist = canonical_text(library.artists[index])
        title = canonical_text(library.titles[index])
        identities.append(
            f"{artist}|{title}" if artist and title else f"track:{library.track_ids[index]}"
        )
        if artist:
            known_artists.append(artist)
    artist_counts = Counter(known_artists)
    return {
        "depth": depth,
        "metadata_key_duplicate_excess": len(metadata) - len(set(metadata)),
        "file_path_duplicate_excess": len(paths) - len(set(paths)),
        "normalized_artist_title_duplicate_excess": len(identities)
        - len(set(identities)),
        "known_unique_artists": len(artist_counts),
        "unknown_artist_count": depth - len(known_artists),
        "largest_artist_count": max(artist_counts.values(), default=0),
        "largest_artist": (
            sorted(artist_counts, key=lambda artist: (-artist_counts[artist], artist))[0]
            if artist_counts
            else None
        ),
    }


def initial_diagnostics(
    library: queue_eval.Library,
    initial_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    diagnostics: list[dict[str, object]] = []
    similarities_by_id: dict[str, np.ndarray] = {}
    for row in initial_rows:
        prompt_id = str(row["id"])
        embedding = np.asarray(row["embedding"], dtype=np.float32)
        similarities = library.embeddings @ embedding
        similarities_by_id[prompt_id] = similarities
        order = result_indices(library, similarities, 50)
        replayed_ids = [int(library.track_ids[index]) for index in order]
        stored_ids = [int(item["track_id"]) for item in row["top_results"]]
        if replayed_ids != stored_ids:
            raise ValueError(f"stored top-50 does not replay exactly for {prompt_id}")
        distribution = row["score_distribution"]
        standard_deviation = float(distribution["std"])
        top_score = float(similarities[order[0]])
        diagnostic: dict[str, object] = {
            "id": prompt_id,
            "kind": row["kind"],
            "prompt": row["prompt"],
            "top_1_score": top_score,
            "top_10_floor": float(similarities[order[9]]),
            "top_50_floor": float(similarities[order[49]]),
            "top_1_z": (
                (top_score - float(distribution["mean"])) / standard_deviation
                if standard_deviation
                else None
            ),
            "top_3": [
                {
                    "track_id": int(library.track_ids[index]),
                    "artist": library.artists[index],
                    "title": library.titles[index],
                    "score": float(similarities[index]),
                }
                for index in order[:3]
            ],
            "crowding": {
                str(depth): crowding_metrics(library, order, depth)
                for depth in (10, 20, 50)
            },
            "fixed_score_threshold_counts": {
                str(threshold): int(np.sum(similarities >= threshold))
                for threshold in (0.10, 0.15, 0.20, 0.25)
            },
        }
        if "target" in row:
            diagnostic["target"] = row["target"]
            diagnostic["target_library_percentile"] = 1.0 - (
                (int(row["target"]["best_rank"]) - 1) / library.count
            )
        diagnostics.append(diagnostic)
    return diagnostics, similarities_by_id


def followup_rows(spec: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in spec["described_track_variants"]:
        for variant, prompt in family["variants"].items():
            rows.append(
                {
                    "id": f"described/{family['family']}/{variant}",
                    "kind": "described_track_followup",
                    "family": family["family"],
                    "variant": variant,
                    "prompt": prompt,
                    "target_track_ids": family["target_track_ids"],
                }
            )
    for group in spec["contrast_groups"]:
        for side in ("left", "right"):
            label = group[f"{side}_label"]
            rows.append(
                {
                    "id": f"contrast/{group['id']}/{label}",
                    "kind": "contrast_followup",
                    "contrast_group": group["id"],
                    "contrast_label": label,
                    "prompt": group[f"{side}_prompt"],
                }
            )
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("follow-up IDs are not unique")
    return rows


def manifest_value(
    args: argparse.Namespace,
    spec_hash: str,
    initial_results_hash: str,
    initial_summary_hash: str,
    db_hash: str,
    model_path: Path,
    model_hash: str,
) -> dict[str, object]:
    return {
        "experiment_status": "follow_up_after_initial_results",
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "track_count": queue_eval.EXPECTED_TRACKS,
            "embedding_dim": queue_eval.EXPECTED_DIM,
            "sqlite_mode": "ro,immutable=1",
        },
        "initial_evidence": {
            "directory": str(args.initial_output.resolve()),
            "prompt_results_sha256": initial_results_hash,
            "summary_sha256": initial_summary_hash,
        },
        "followup_spec": {
            "path": str(args.spec.resolve()),
            "sha256": spec_hash,
        },
        "model": {
            "name": "sander-wood/clamp3",
            "checkpoint": str(model_path),
            "checkpoint_sha256": model_hash,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }


def validate_or_create_manifest(
    path: Path,
    expected: dict[str, object],
    force: bool,
) -> None:
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError("existing output manifest does not match current inputs")
    else:
        write_json_atomic(path, expected)


def described_sensitivity(
    initial_rows_by_id: dict[str, dict[str, object]],
    followup_by_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    families: list[dict[str, object]] = []
    ranks_by_variant: dict[str, list[int]] = {"original": []}
    for variant in ("terse", "audio_only", "paraphrase"):
        ranks_by_variant[variant] = []
    for original_id, original in initial_rows_by_id.items():
        if original["kind"] != "described_track":
            continue
        family = original_id.removeprefix("described/")
        original_embedding = np.asarray(original["embedding"], dtype=np.float32)
        original_ids = [
            int(item["track_id"]) for item in original["top_results"][:50]
        ]
        row: dict[str, object] = {
            "family": family,
            "original": {
                "prompt": original["prompt"],
                "best_target_rank": int(original["target"]["best_rank"]),
                "best_target_score": float(original["target"]["best_score"]),
            },
            "variants": {},
        }
        ranks_by_variant["original"].append(int(original["target"]["best_rank"]))
        for variant in ("terse", "audio_only", "paraphrase"):
            followup = followup_by_id[f"described/{family}/{variant}"]
            embedding = np.asarray(followup["embedding"], dtype=np.float32)
            ids = [int(item["track_id"]) for item in followup["top_results"][:50]]
            target_rank = int(followup["target"]["best_rank"])
            ranks_by_variant[variant].append(target_rank)
            row["variants"][variant] = {
                "prompt": followup["prompt"],
                "best_target_rank": target_rank,
                "best_target_score": float(followup["target"]["best_score"]),
                "embedding_cosine_to_original": float(embedding @ original_embedding),
                "top_50_to_original": overlap(ids, original_ids),
                "top_3": followup["top_results"][:3],
            }
        families.append(row)
    aggregates: dict[str, object] = {}
    original_ranks = np.asarray(ranks_by_variant["original"], dtype=np.int64)
    for variant, values in ranks_by_variant.items():
        ranks = np.asarray(values, dtype=np.int64)
        aggregate: dict[str, object] = {
            "median_best_target_rank": float(np.median(ranks)),
            "geometric_mean_best_target_rank": float(np.exp(np.mean(np.log(ranks)))),
            "within_top_10": int(np.sum(ranks <= 10)),
            "within_top_50": int(np.sum(ranks <= 50)),
            "within_top_100": int(np.sum(ranks <= 100)),
            "within_top_1000": int(np.sum(ranks <= 1000)),
            "improved_vs_original": (
                int(np.sum(ranks < original_ranks)) if variant != "original" else None
            ),
            "worsened_vs_original": (
                int(np.sum(ranks > original_ranks)) if variant != "original" else None
            ),
        }
        if variant != "original":
            variant_rows = [row["variants"][variant] for row in families]
            overlap_counts = np.asarray(
                [int(row["top_50_to_original"]["intersection"]) for row in variant_rows]
            )
            aggregate.update(
                {
                    "median_embedding_cosine_to_original": float(
                        np.median(
                            [
                                float(row["embedding_cosine_to_original"])
                                for row in variant_rows
                            ]
                        )
                    ),
                    "median_top_50_overlap_count": float(np.median(overlap_counts)),
                    "minimum_top_50_overlap_count": int(np.min(overlap_counts)),
                    "maximum_top_50_overlap_count": int(np.max(overlap_counts)),
                }
            )
        aggregates[variant] = aggregate
    return {"aggregates": aggregates, "families": families}


def exact_repeat_summary(
    initial_rows: list[dict[str, object]],
    followup_rows_by_id: dict[str, dict[str, object]],
) -> dict[str, object]:
    initial_by_prompt = {str(row["prompt"]): row for row in initial_rows}
    repeated: list[dict[str, object]] = []
    for followup in followup_rows_by_id.values():
        initial = initial_by_prompt.get(str(followup["prompt"]))
        if initial is None:
            continue
        embeddings_exact = initial["embedding"] == followup["embedding"]
        initial_ids = [int(row["track_id"]) for row in initial["top_results"]]
        followup_ids = [int(row["track_id"]) for row in followup["top_results"]]
        top_50_exact = initial_ids == followup_ids
        if not embeddings_exact or not top_50_exact:
            raise ValueError(f"exact repeated prompt changed: {followup['id']}")
        repeated.append(
            {
                "initial_id": initial["id"],
                "followup_id": followup["id"],
                "embedding_exact": embeddings_exact,
                "top_50_exact": top_50_exact,
            }
        )
    return {"count": len(repeated), "repeats": repeated}


def contrast_summary(
    spec: dict[str, object],
    followup_by_id: dict[str, dict[str, object]],
    library: queue_eval.Library,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in spec["contrast_groups"]:
        left_id = f"contrast/{group['id']}/{group['left_label']}"
        right_id = f"contrast/{group['id']}/{group['right_label']}"
        left = followup_by_id[left_id]
        right = followup_by_id[right_id]
        left_embedding = np.asarray(left["embedding"], dtype=np.float32)
        right_embedding = np.asarray(right["embedding"], dtype=np.float32)
        left_scores = library.embeddings @ left_embedding
        right_scores = library.embeddings @ right_embedding
        left_order = result_indices(library, left_scores, 50)
        right_order = result_indices(library, right_scores, 50)
        row: dict[str, object] = {
            "group": group["id"],
            "left_label": group["left_label"],
            "right_label": group["right_label"],
            "embedding_cosine": float(left_embedding @ right_embedding),
            "left_top_10_mean_preference": float(
                np.mean(left_scores[left_order[:10]] - right_scores[left_order[:10]])
            ),
            "right_top_10_mean_preference": float(
                np.mean(right_scores[right_order[:10]] - left_scores[right_order[:10]])
            ),
        }
        for depth in (10, 20, 50):
            row[f"top_{depth}"] = overlap(
                library.track_ids[left_order[:depth]],
                library.track_ids[right_order[:depth]],
            )
        rows.append(row)
    return rows


def score_and_crowding_summary(
    diagnostics: list[dict[str, object]],
) -> dict[str, object]:
    by_kind: dict[str, list[dict[str, object]]] = {}
    for row in diagnostics:
        by_kind.setdefault(str(row["kind"]), []).append(row)
    score_kinds: dict[str, object] = {}
    for kind, rows in by_kind.items():
        top_1 = np.asarray([float(row["top_1_score"]) for row in rows])
        top_50 = np.asarray([float(row["top_50_floor"]) for row in rows])
        score_kinds[kind] = {
            "count": len(rows),
            "top_1_min": float(np.min(top_1)),
            "top_1_median": float(np.median(top_1)),
            "top_1_max": float(np.max(top_1)),
            "top_50_floor_min": float(np.min(top_50)),
            "top_50_floor_median": float(np.median(top_50)),
            "top_50_floor_max": float(np.max(top_50)),
        }
    crowding_totals: dict[str, object] = {}
    for depth in (10, 20, 50):
        depth_rows = [row["crowding"][str(depth)] for row in diagnostics]
        crowding_totals[str(depth)] = {
            "prompt_count_with_normalized_duplicate": sum(
                int(value["normalized_artist_title_duplicate_excess"]) > 0
                for value in depth_rows
            ),
            "normalized_duplicate_excess_total": sum(
                int(value["normalized_artist_title_duplicate_excess"])
                for value in depth_rows
            ),
            "metadata_duplicate_excess_total": sum(
                int(value["metadata_key_duplicate_excess"]) for value in depth_rows
            ),
            "path_duplicate_excess_total": sum(
                int(value["file_path_duplicate_excess"]) for value in depth_rows
            ),
            "median_unique_known_artists": float(
                np.median([int(value["known_unique_artists"]) for value in depth_rows])
            ),
            "maximum_single_artist_count": max(
                int(value["largest_artist_count"]) for value in depth_rows
            ),
        }
    return {"scores_by_kind": score_kinds, "crowding": crowding_totals}


def hub_summary(
    initial_rows: list[dict[str, object]],
    top_k: int = 10,
) -> list[dict[str, object]]:
    counts: Counter[int] = Counter()
    labels: dict[int, tuple[str | None, str | None]] = {}
    prompt_ids: dict[int, list[str]] = {}
    for row in initial_rows:
        for result in row["top_results"][:top_k]:
            track_id = int(result["track_id"])
            counts[track_id] += 1
            labels[track_id] = (result["artist"], result["title"])
            prompt_ids.setdefault(track_id, []).append(str(row["id"]))
    return [
        {
            "track_id": track_id,
            "artist": labels[track_id][0],
            "title": labels[track_id][1],
            "top_10_occurrences": count,
            "prompt_ids": prompt_ids[track_id],
        }
        for track_id, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )[:30]
    ]


def qualitative_markdown(
    path: Path,
    initial_rows: list[dict[str, object]],
    diagnostics_by_id: dict[str, dict[str, object]],
) -> None:
    lines = [
        "# Frozen Initial Text Rankings",
        "",
        "This is generated evidence. It includes all 55 original prompts and does not",
        "assign a listening verdict from metadata alone.",
        "",
    ]
    for row in initial_rows:
        diagnostic = diagnostics_by_id[str(row["id"])]
        target = ""
        if "target" in row:
            target = f"; target rank {row['target']['best_rank']}"
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"> {row['prompt']}",
                "",
                f"Top score `{diagnostic['top_1_score']:.6f}`; top-50 floor "
                f"`{diagnostic['top_50_floor']:.6f}`{target}.",
                "",
            ]
        )
        for result in row["top_results"][:10]:
            artist = result["artist"] or "?"
            title = result["title"] or "?"
            lines.append(
                f"{result['rank']}. {artist} - {title} (`{result['score']:.6f}`)"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--initial-output", type=Path, default=DEFAULT_INITIAL_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 50:
        raise ValueError("--top-k must be at least 50")
    initial_results_path = args.initial_output / "prompt-results.jsonl"
    initial_summary_path = args.initial_output / "summary.json"
    initial_results_hash = sha256_file(initial_results_path)
    initial_summary_hash = sha256_file(initial_summary_path)
    if initial_results_hash != EXPECTED_INITIAL_RESULTS_SHA256:
        raise ValueError("initial prompt-results.jsonl SHA-256 mismatch")
    if initial_summary_hash != EXPECTED_INITIAL_SUMMARY_SHA256:
        raise ValueError("initial summary.json SHA-256 mismatch")

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec_hash = sha256_file(args.spec)
    initial_rows = read_jsonl(initial_results_path)
    if len(initial_rows) != 55:
        raise ValueError(f"expected 55 initial prompts, found {len(initial_rows)}")
    library, db_hash = queue_eval.load_library(args.db, verify_hash=True)
    diagnostics, _ = initial_diagnostics(library, initial_rows)

    model_path = initial_eval.resolve_checkpoint()
    model_hash = sha256_file(model_path)
    expected_manifest = manifest_value(
        args,
        spec_hash,
        initial_results_hash,
        initial_summary_hash,
        db_hash,
        model_path,
        model_hash,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    results_path = args.output / "followup-results.jsonl"
    if args.force:
        for path in (
            results_path,
            args.output / "summary.json",
            args.output / "qualitative-initial.md",
            args.output / "manifest.json",
        ):
            if path.exists():
                path.unlink()
    validate_or_create_manifest(
        args.output / "manifest.json", expected_manifest, force=args.force
    )

    prompts = followup_rows(spec)
    completed = {
        str(row["id"]): row for row in read_jsonl(results_path)
    } if results_path.exists() else {}
    for row in prompts:
        existing = completed.get(str(row["id"]))
        if existing is not None and (
            existing.get("prompt") != row["prompt"]
            or existing.get("target_track_ids") != row.get("target_track_ids")
        ):
            raise ValueError(f"stale checkpoint record for {row['id']}")

    generator = CLaMP3EmbeddingGenerator(fp16=False)
    try:
        for position, prompt in enumerate(prompts, start=1):
            prompt_id = str(prompt["id"])
            if prompt_id in completed:
                print(f"{position}/{len(prompts)} {prompt_id}: resumed", file=sys.stderr)
                continue
            started = time.perf_counter()
            record = initial_eval.evaluate_prompt(
                generator, library, prompt, args.top_k
            )
            record["followup_status"] = "authored_after_initial_rankings"
            record["wall_ms_including_ranking"] = (
                time.perf_counter() - started
            ) * 1000.0
            append_jsonl(results_path, record)
            completed[prompt_id] = record
            print(
                f"{position}/{len(prompts)} {prompt_id}: "
                f"{record['inference_ms']:.1f} ms text",
                file=sys.stderr,
            )
    finally:
        generator.unload_models()

    ordered_followup = {str(row["id"]): completed[str(row["id"])] for row in prompts}
    initial_by_id = {str(row["id"]): row for row in initial_rows}
    diagnostics_by_id = {str(row["id"]): row for row in diagnostics}
    summary = {
        "provenance": expected_manifest,
        "initial_prompt_count": len(initial_rows),
        "followup_prompt_count": len(prompts),
        "initial_top_50_replay": "exact_track_id_match_for_all_55",
        "initial_analysis": score_and_crowding_summary(diagnostics),
        "initial_prompt_diagnostics": diagnostics,
        "initial_top_10_recurring_tracks": hub_summary(initial_rows),
        "exact_repeated_prompts": exact_repeat_summary(
            initial_rows, ordered_followup
        ),
        "described_prompt_sensitivity": described_sensitivity(
            initial_by_id, ordered_followup
        ),
        "contrast_pairs": contrast_summary(spec, ordered_followup, library),
        "followup_inference_ms": {
            "median": float(
                np.median(
                    [float(row["inference_ms"]) for row in ordered_followup.values()]
                )
            ),
            "p95": float(
                np.percentile(
                    [float(row["inference_ms"]) for row in ordered_followup.values()],
                    95,
                )
            ),
        },
    }
    write_json_atomic(args.output / "summary.json", summary)
    qualitative_markdown(
        args.output / "qualitative-initial.md",
        initial_rows,
        diagnostics_by_id,
    )
    print(f"Complete. Results: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
