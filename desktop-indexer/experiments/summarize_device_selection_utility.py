#!/usr/bin/env python3
"""Summarize real-phone selector output without pretending geometry is listening quality."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_queue_eval as queue_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "2026-07-07T223308+0300_qv7706c3mq"
    / "embeddings.db"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualitative", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_artist(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip().casefold()
    return result or None


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "min": None, "p50": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.median(array)),
        "max": float(array.max()),
    }


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def canonical_default_case(case_id: str) -> str | None:
    exact = {
        "closest": "closest",
        "mmr": "mmr",
        "dpp": "dpp",
        "graph_explorer": "graph_explorer",
        "mmr_seed_interpolation": "mmr_seed_interpolation",
        "mmr_momentum": "mmr_momentum",
        "uniform_shuffle": "uniform_shuffle",
        "v2_closest_default": "closest",
        "v2_uniform_shuffle_default": "uniform_shuffle",
    }
    if case_id in exact:
        return exact[case_id]
    prefixes = {
        "v2_mmr_default__": "mmr",
        "v2_dpp_full_default__": "dpp",
        "v2_graph_default__": "graph_explorer",
        "v2_drift_seed_default__": "mmr_seed_interpolation",
        "v2_drift_momentum_default__": "mmr_momentum",
    }
    return next(
        (canonical for prefix, canonical in prefixes.items() if case_id.startswith(prefix)),
        None,
    )


def first_repeat_runs(report: dict[str, object]) -> list[dict[str, object]]:
    return [
        run
        for run in report.get("selectionRuns", [])
        if int(run["repeat"]) == 1
    ]


def run_metrics(
    run: dict[str, object],
    library: queue_eval.Library,
    id_to_position: dict[int, int],
) -> dict[str, object]:
    track_rows = run["tracks"]
    positions = np.asarray(
        [id_to_position[int(row["trackId"])] for row in track_rows],
        dtype=np.int64,
    )
    vectors = library.embeddings[positions]
    seed_position = id_to_position[int(run["seedTrackId"])]
    seed_cosines = vectors @ library.embeddings[seed_position]
    adjacent = (
        np.sum(vectors[:-1] * vectors[1:], axis=1, dtype=np.float32)
        if vectors.shape[0] > 1
        else np.empty(0, dtype=np.float32)
    )
    if vectors.shape[0] > 1:
        pairwise = vectors @ vectors.T
        pairwise_values = pairwise[np.triu_indices(vectors.shape[0], k=1)]
    else:
        pairwise_values = np.empty(0, dtype=np.float32)
    known_artists = [
        artist
        for artist in (normalized_artist(row.get("artist")) for row in track_rows)
        if artist is not None
    ]
    return {
        "seed_track_id": int(run["seedTrackId"]),
        "case_id": str(run["caseId"]),
        "track_ids": [int(row["trackId"]) for row in track_rows],
        "count": len(track_rows),
        "elapsed_ms": int(run["elapsedMs"]),
        "mean_seed_cosine": float(seed_cosines.mean()),
        "minimum_seed_cosine": float(seed_cosines.min()),
        "final_seed_cosine": float(seed_cosines[-1]),
        "mean_adjacent_cosine": float(adjacent.mean()) if adjacent.size else None,
        "minimum_adjacent_cosine": float(adjacent.min()) if adjacent.size else None,
        "mean_pairwise_cosine": (
            float(pairwise_values.mean()) if pairwise_values.size else None
        ),
        "unique_known_artist_credits": len(set(known_artists)),
        "known_artist_credit_count": len(known_artists),
        "final_global_seed_rank": track_rows[-1].get("seedRank") if track_rows else None,
        "matcher_disagreement_count": int(run["matcherDisagreementCount"]),
        "matcher_missing_count": int(run["resolvedCount"]) - int(run["matcherResolvedCount"]),
        "inactive_result_count": int(run["inactiveResultCount"]),
    }


def aggregate_modes(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metrics:
        by_case[str(row["case_id"])].append(row)
    result: dict[str, object] = {}
    for case_id, rows in sorted(by_case.items()):
        result[case_id] = {
            "seed_count": len(rows),
            "elapsed_ms": numeric_summary(float(row["elapsed_ms"]) for row in rows),
            "mean_seed_cosine": numeric_summary(
                float(row["mean_seed_cosine"]) for row in rows
            ),
            "minimum_seed_cosine": numeric_summary(
                float(row["minimum_seed_cosine"]) for row in rows
            ),
            "final_seed_cosine": numeric_summary(
                float(row["final_seed_cosine"]) for row in rows
            ),
            "mean_adjacent_cosine": numeric_summary(
                float(row["mean_adjacent_cosine"])
                for row in rows
                if row["mean_adjacent_cosine"] is not None
            ),
            "minimum_adjacent_cosine": numeric_summary(
                float(row["minimum_adjacent_cosine"])
                for row in rows
                if row["minimum_adjacent_cosine"] is not None
            ),
            "mean_pairwise_cosine": numeric_summary(
                float(row["mean_pairwise_cosine"])
                for row in rows
                if row["mean_pairwise_cosine"] is not None
            ),
            "unique_known_artist_credits": numeric_summary(
                float(row["unique_known_artist_credits"]) for row in rows
            ),
            "matcher_disagreement_count": sum(
                int(row["matcher_disagreement_count"]) for row in rows
            ),
            "matcher_missing_count": sum(
                int(row["matcher_missing_count"]) for row in rows
            ),
        }
    return result


def repeat_evidence(report: dict[str, object]) -> dict[str, object]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    for run in report.get("selectionRuns", []):
        grouped[(int(run["seedTrackId"]), str(run["caseId"]))].append(run)
    groups: list[dict[str, object]] = []
    for (seed_track_id, case_id), runs in sorted(grouped.items()):
        ordered = sorted(runs, key=lambda run: int(run["repeat"]))
        track_orders = [
            tuple(int(row["trackId"]) for row in run["tracks"])
            for run in ordered
        ]
        fingerprints = [run.get("resultFingerprint") for run in ordered]
        groups.append(
            {
                "seed_track_id": seed_track_id,
                "case_id": case_id,
                "repeat_count": len(ordered),
                "unique_track_orders": len(set(track_orders)),
                "unique_fingerprints": len(set(fingerprints)),
                "elapsed_ms": [int(run["elapsedMs"]) for run in ordered],
            }
        )
    return {
        "groups": groups,
        "all_exact": all(
            row["repeat_count"] >= 2
            and row["unique_track_orders"] == 1
            and row["unique_fingerprints"] == 1
            for row in groups
        ),
    }


def overlap_evidence(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metrics:
        canonical = canonical_default_case(str(row["case_id"]))
        if canonical is not None:
            by_seed[int(row["seed_track_id"])][canonical] = row
    pairs = [
        ("closest", "mmr"),
        ("mmr", "dpp"),
        ("mmr", "graph_explorer"),
        ("mmr", "mmr_seed_interpolation"),
        ("mmr", "mmr_momentum"),
        ("dpp", "graph_explorer"),
        ("graph_explorer", "uniform_shuffle"),
    ]
    result: dict[str, object] = {}
    for left, right in pairs:
        values: list[float] = []
        intersections: list[int] = []
        for modes in by_seed.values():
            if left not in modes or right not in modes:
                continue
            left_ids = modes[left]["track_ids"]
            right_ids = modes[right]["track_ids"]
            values.append(jaccard(left_ids, right_ids))
            intersections.append(len(set(left_ids) & set(right_ids)))
        result[f"{left}__{right}"] = {
            "jaccard": numeric_summary(values),
            "intersection_of_30": numeric_summary(intersections),
        }
    return result


def breadth_evidence(metrics: Sequence[dict[str, object]]) -> dict[str, object]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metrics:
        canonical = canonical_default_case(str(row["case_id"]))
        if canonical is not None:
            by_seed[int(row["seed_track_id"])][canonical] = row
    comparisons: dict[str, object] = {}
    for candidate in ("dpp", "graph_explorer", "mmr_seed_interpolation", "mmr_momentum"):
        rows = []
        for seed_track_id, modes in sorted(by_seed.items()):
            if "mmr" not in modes or candidate not in modes:
                continue
            baseline = modes["mmr"]
            alternate = modes[candidate]
            alternate_pairwise = alternate["mean_pairwise_cosine"]
            baseline_pairwise = baseline["mean_pairwise_cosine"]
            rows.append(
                {
                    "seed_track_id": seed_track_id,
                    "mean_seed_cosine_delta": float(alternate["mean_seed_cosine"])
                    - float(baseline["mean_seed_cosine"]),
                    "mean_pairwise_cosine_delta": (
                        float(alternate_pairwise) - float(baseline_pairwise)
                        if alternate_pairwise is not None and baseline_pairwise is not None
                        else None
                    ),
                    "artist_credit_delta": int(alternate["unique_known_artist_credits"])
                    - int(baseline["unique_known_artist_credits"]),
                }
            )
        comparisons[candidate] = {
            "per_seed": rows,
            "broader_by_both_cosines": sum(
                row["mean_seed_cosine_delta"] < 0.0
                and row["mean_pairwise_cosine_delta"] is not None
                and row["mean_pairwise_cosine_delta"] < 0.0
                for row in rows
            ),
            "pairwise_evaluable_seed_count": sum(
                row["mean_pairwise_cosine_delta"] is not None for row in rows
            ),
            "seed_count": len(rows),
        }
    return comparisons


def qualitative_markdown(
    report: dict[str, object],
    metrics: Sequence[dict[str, object]],
) -> str:
    def metric(value: object) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    metrics_by_key = {
        (int(row["seed_track_id"]), str(row["case_id"])): row for row in metrics
    }
    runs_by_seed: dict[int, list[dict[str, object]]] = defaultdict(list)
    for run in first_repeat_runs(report):
        runs_by_seed[int(run["seedTrackId"])].append(run)
    lines = [
        "# Device Selection Utility Packet",
        "",
        "These are the complete, unedited queues produced on the phone. Geometry describes",
        "reach, redundancy, and continuity; it does not by itself certify listening quality.",
        "",
    ]
    for seed_track_id, runs in sorted(runs_by_seed.items()):
        seed = runs[0].get("seed") or {}
        lines.extend(
            [
                f"## {seed.get('artist') or '?'} - {seed.get('title') or '?'}",
                "",
                f"Embedding track ID: `{seed_track_id}`",
                "",
            ]
        )
        for run in runs:
            case_id = str(run["caseId"])
            row = metrics_by_key[(seed_track_id, case_id)]
            lines.extend(
                [
                    f"### {case_id}",
                    "",
                    (
                        f"mean seed `{metric(row['mean_seed_cosine'])}`; minimum seed "
                        f"`{metric(row['minimum_seed_cosine'])}`; mean pairwise "
                        f"`{metric(row['mean_pairwise_cosine'])}`; mean adjacent "
                        f"`{metric(row['mean_adjacent_cosine'])}`; artist credits "
                        f"`{row['unique_known_artist_credits']}`; phone "
                        f"`{row['elapsed_ms']} ms`"
                    ),
                    "",
                ]
            )
            for track in run["tracks"]:
                artist = track.get("artist") or "?"
                title = track.get("title") or "?"
                lines.append(
                    f"{track['rank']}. {artist} - {title} "
                    f"[seed {float(track['similarityToSeed']):.4f}]"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE":
        raise ValueError("phone report is not complete")
    database_sha256 = sha256_file(args.database)
    expected_database_sha256 = report.get("generation", {}).get("databaseSha256")
    if expected_database_sha256 is not None and database_sha256 != expected_database_sha256:
        raise ValueError(
            "report/database mismatch: "
            f"report expects {expected_database_sha256}, got {database_sha256}"
        )
    library, _ = queue_eval.load_library(args.database, verify_hash=False)
    id_to_position = {
        int(track_id): position
        for position, track_id in enumerate(library.track_ids)
    }
    metrics = [
        run_metrics(run, library, id_to_position)
        for run in first_repeat_runs(report)
    ]
    output = {
        "schema_version": 1,
        "phone_run_id": report["runId"],
        "report_sha256": sha256_file(args.report),
        "database_sha256": database_sha256,
        "seed_count": len({int(row["seed_track_id"]) for row in metrics}),
        "mode_count": len({str(row["case_id"]) for row in metrics}),
        "repeat_evidence": repeat_evidence(report),
        "mode_metrics": aggregate_modes(metrics),
        "mode_overlap": overlap_evidence(metrics),
        "breadth_vs_mmr": breadth_evidence(metrics),
        "per_seed_mode_metrics": metrics,
        "interpretation_boundary": (
            "These metrics establish deterministic behavior, geometric reach, redundancy, "
            "continuity, and distinctness. They do not establish subjective listening quality."
        ),
    }
    atomic_text(args.output, json.dumps(output, indent=2, sort_keys=True) + "\n")
    atomic_text(args.qualitative, qualitative_markdown(report, metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
