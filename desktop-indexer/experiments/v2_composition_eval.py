#!/usr/bin/env python3
"""Evaluate explicit multi-anchor query semantics on the immutable phone library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import v2_queue_eval as queue_eval


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TEXT_RESULTS = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "text"
    / "prompt-results.jsonl"
)
DEFAULT_OUTPUT = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "composition"
)
TOP_K = 50


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    label: str
    embedding: np.ndarray
    weight: float


# Frozen before composed rankings were generated. Song IDs come from the described-track
# cohort, and text IDs come from the preregistered operator-anchor list.
PRIMARY_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "hallucinogen_dark_ambient",
        "operators": ("balance", "direction"),
        "anchors": (("song:13384", 0.65), ("text:anchor/00", 0.35)),
    },
    {
        "id": "nusrat_broken_beat",
        "operators": ("balance", "direction"),
        "anchors": (("song:209", 0.65), ("text:anchor/04", 0.35)),
    },
    {
        "id": "burial_female_vocals",
        "operators": ("balance", "direction"),
        "anchors": (("song:5831", 0.7), ("text:anchor/06", 0.3)),
    },
    {
        "id": "tool_more_percussion",
        "operators": ("balance", "direction"),
        "anchors": (("song:31830", 0.7), ("text:anchor/03", 0.3)),
    },
    {
        "id": "tool_less_guitar",
        "operators": ("balance", "direction"),
        "anchors": (("song:31830", 0.8), ("text:anchor/02", -0.2)),
    },
    {
        "id": "hallucinogen_or_nusrat",
        "operators": ("balance", "direction", "either"),
        "anchors": (("song:13384", 0.5), ("song:209", 0.5)),
    },
    {
        "id": "aphex_or_dgary",
        "operators": ("balance", "direction", "either"),
        "anchors": (("song:2682", 0.5), ("song:7838", 0.5)),
    },
    {
        "id": "dark_ambient_or_devotional",
        "operators": ("balance", "direction", "either"),
        "anchors": (("text:anchor/00", 0.5), ("text:anchor/01", 0.5)),
    },
    {
        "id": "instrumental_female_vocals_contradiction",
        "operators": ("balance", "direction"),
        "anchors": (("text:anchor/06", 0.5), ("text:anchor/07", 0.5)),
    },
    {
        "id": "exact_signed_contradiction",
        "operators": ("balance", "direction"),
        "anchors": (("song:13384", 0.5), ("song:13384", -0.5)),
    },
    {
        "id": "hallucinogen_less_ambient_more_broken",
        "operators": ("analogy",),
        "anchors": (
            ("song:13384", 1.0),
            ("text:anchor/00", -1.0),
            ("text:anchor/04", 1.0),
        ),
    },
    {
        "id": "tool_less_guitar_more_percussion",
        "operators": ("analogy",),
        "anchors": (
            ("song:31830", 1.0),
            ("text:anchor/02", -1.0),
            ("text:anchor/03", 1.0),
        ),
    },
)


# Registered after the primary run; see COMPOSITION_EXPERIMENTS.md F01.
FOLLOWUP_WEIGHT_CASES: tuple[dict[str, str], ...] = (
    {
        "id": "hallucinogen_dark_ambient",
        "first": "song:13384",
        "second": "text:anchor/00",
    },
    {
        "id": "nusrat_broken_beat",
        "first": "song:209",
        "second": "text:anchor/04",
    },
    {
        "id": "burial_female_vocals",
        "first": "song:5831",
        "second": "text:anchor/06",
    },
    {
        "id": "hallucinogen_nusrat",
        "first": "song:13384",
        "second": "song:209",
    },
)


# Registered after F01; see COMPOSITION_EXPERIMENTS.md F02.
FOLLOWUP_REFINE_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "hallucinogen_dark_ambient",
        "primary": "song:13384",
        "secondary": "text:anchor/00",
        "secondary_sign": 1.0,
    },
    {
        "id": "nusrat_broken_beat",
        "primary": "song:209",
        "secondary": "text:anchor/04",
        "secondary_sign": 1.0,
    },
    {
        "id": "burial_female_vocals",
        "primary": "song:5831",
        "secondary": "text:anchor/06",
        "secondary_sign": 1.0,
    },
    {
        "id": "tool_more_percussion",
        "primary": "song:31830",
        "secondary": "text:anchor/03",
        "secondary_sign": 1.0,
    },
    {
        "id": "tool_less_guitar",
        "primary": "song:31830",
        "secondary": "text:anchor/02",
        "secondary_sign": -1.0,
    },
)
REFINE_FRACTIONS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.05)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text_embeddings(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    result: dict[str, tuple[str, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_id = str(row["id"])
            embedding = np.asarray(row["embedding"], dtype=np.float32)
            if embedding.shape != (queue_eval.EXPECTED_DIM,):
                raise ValueError(f"text row {line_number} has shape {embedding.shape}")
            result[prompt_id] = (str(row["prompt"]), embedding)
    return result


def track_index_by_id(library: queue_eval.Library) -> dict[int, int]:
    return {int(track_id): index for index, track_id in enumerate(library.track_ids)}


def resolve_anchor(
    spec: tuple[str, float],
    library: queue_eval.Library,
    by_track_id: dict[int, int],
    text_embeddings: dict[str, tuple[str, np.ndarray]],
) -> Anchor:
    reference, weight = spec
    kind, value = reference.split(":", 1)
    if kind == "song":
        track_id = int(value)
        index = by_track_id[track_id]
        label = f"{library.artists[index] or '?'} - {library.titles[index] or '?'}"
        embedding = library.embeddings[index]
    elif kind == "text":
        label, embedding = text_embeddings[value]
    else:
        raise ValueError(f"unknown anchor kind: {kind}")
    return Anchor(reference, label, embedding, float(weight))


def similarity_percentiles(
    similarities: np.ndarray, track_ids: np.ndarray
) -> np.ndarray:
    """Production-shaped ordinal percentiles with an explicit track-ID tie break."""
    order = np.lexsort((track_ids, similarities))
    ranks = np.empty(similarities.size, dtype=np.int64)
    ranks[order] = np.arange(1, similarities.size + 1, dtype=np.int64)
    return ranks.astype(np.float64) / similarities.size


def per_anchor_data(
    library: queue_eval.Library, anchors: Sequence[Anchor]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    similarities: list[np.ndarray] = []
    percentiles: list[np.ndarray] = []
    for anchor in anchors:
        sims = library.embeddings @ anchor.embedding
        similarities.append(sims)
        ranked = -sims if anchor.weight < 0 else sims
        percentiles.append(similarity_percentiles(ranked, library.track_ids))
    return similarities, percentiles


def normalized_signed_centroid(anchors: Sequence[Anchor]) -> np.ndarray | None:
    vector = np.zeros_like(anchors[0].embedding, dtype=np.float64)
    total_abs = sum(abs(anchor.weight) for anchor in anchors)
    if total_abs <= 1e-12:
        return None
    for anchor in anchors:
        vector += anchor.embedding.astype(np.float64) * (anchor.weight / total_abs)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-7:
        return None
    return (vector / norm).astype(np.float32)


def operator_scores(
    operator: str,
    library: queue_eval.Library,
    anchors: Sequence[Anchor],
    percentiles: Sequence[np.ndarray],
) -> tuple[np.ndarray | None, str | None]:
    if operator == "balance":
        total_abs = sum(abs(anchor.weight) for anchor in anchors)
        if total_abs <= 1e-12:
            return None, "all weights are zero"
        log_score = np.zeros(library.count, dtype=np.float64)
        for anchor, percentile in zip(anchors, percentiles):
            norm_weight = abs(anchor.weight) / total_abs
            log_score += norm_weight * np.log(np.maximum(percentile, 1.0 / library.count))
        return np.exp(log_score), None

    if operator == "either":
        if any(anchor.weight < 0 for anchor in anchors):
            return None, "Either does not accept negative anchors"
        positive = [anchor for anchor in anchors if anchor.weight > 0]
        if not positive:
            return None, "Either needs a positive anchor"
        total = sum(anchor.weight for anchor in positive)
        score = np.full(library.count, -np.inf, dtype=np.float64)
        for anchor, percentile in zip(anchors, percentiles):
            if anchor.weight > 0:
                score = np.maximum(score, (anchor.weight / total) * percentile)
        return score, None

    if operator in {"direction", "analogy"}:
        if operator == "analogy" and len(anchors) != 3:
            return None, "Analogy requires exactly three anchors"
        direction = normalized_signed_centroid(anchors)
        if direction is None:
            return None, "signed anchors cancel to a zero direction"
        return library.embeddings @ direction, None

    raise ValueError(operator)


def ranked_indices(
    scores: np.ndarray, track_ids: np.ndarray, excluded_ids: set[int], count: int
) -> np.ndarray:
    eligible = ~np.isin(track_ids, np.fromiter(excluded_ids, dtype=np.int64))
    indices = np.flatnonzero(eligible)
    order = np.lexsort((track_ids[indices], -scores[indices]))
    return indices[order[:count]]


def branch_coverage(
    chosen: np.ndarray,
    percentiles: Sequence[np.ndarray],
    depth: int = 100,
) -> list[int]:
    threshold = 1.0 - (depth / len(percentiles[0]))
    return [int(np.sum(percentile[chosen] > threshold)) for percentile in percentiles]


def result_record(
    library: queue_eval.Library,
    case_id: str,
    operator: str,
    anchors: Sequence[Anchor],
) -> dict[str, object]:
    started = time.perf_counter()
    similarities, percentiles = per_anchor_data(library, anchors)
    scores, rejection = operator_scores(operator, library, anchors, percentiles)
    if rejection is not None or scores is None:
        return {
            "case_id": case_id,
            "operator": operator,
            "anchors": anchor_records(anchors),
            "status": "rejected",
            "reason": rejection,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    excluded = {
        int(anchor.anchor_id.split(":", 1)[1])
        for anchor in anchors
        if anchor.anchor_id.startswith("song:")
    }
    chosen = ranked_indices(scores, library.track_ids, excluded, TOP_K)
    rows: list[dict[str, object]] = []
    for rank, index in enumerate(chosen, start=1):
        rows.append(
            {
                "rank": rank,
                "track_id": int(library.track_ids[index]),
                "artist": library.artists[index],
                "album": library.albums[index],
                "title": library.titles[index],
                "score": float(scores[index]),
                "anchor_cosines": [float(values[index]) for values in similarities],
                "effective_anchor_percentiles": [
                    float(values[index]) for values in percentiles
                ],
            }
        )

    top20 = chosen[:20]
    effective = np.stack([values[top20] for values in percentiles], axis=1)
    return {
        "case_id": case_id,
        "operator": operator,
        "anchors": anchor_records(anchors),
        "status": "ranked",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "top20_mean_effective_percentile_by_anchor": [
            float(value) for value in np.mean(effective, axis=0)
        ],
        "top20_min_effective_percentile_by_anchor": [
            float(value) for value in np.min(effective, axis=0)
        ],
        "top20_anchor_top100_coverage": branch_coverage(top20, percentiles),
        "top_results": rows,
    }


def weight_sweep_record(
    library: queue_eval.Library,
    case: dict[str, str],
    operator: str,
    by_track_id: dict[int, int],
    text_embeddings: dict[str, tuple[str, np.ndarray]],
) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    previous_ids: set[int] | None = None
    for second_tenths in range(11):
        second_weight = second_tenths / 10.0
        anchors = [
            resolve_anchor(
                (case["first"], 1.0 - second_weight),
                library,
                by_track_id,
                text_embeddings,
            ),
            resolve_anchor(
                (case["second"], second_weight),
                library,
                by_track_id,
                text_embeddings,
            ),
        ]
        similarities, percentiles = per_anchor_data(library, anchors)
        scores, error = operator_scores(operator, library, anchors, percentiles)
        if error is not None or scores is None:
            raise AssertionError(f"unexpected sweep rejection: {case['id']} {operator} {error}")
        excluded = {
            int(anchor.anchor_id.split(":", 1)[1])
            for anchor in anchors
            if anchor.anchor_id.startswith("song:")
        }
        chosen = ranked_indices(scores, library.track_ids, excluded, 20)
        chosen_ids = {int(library.track_ids[index]) for index in chosen}
        effective = np.stack([values[chosen] for values in percentiles], axis=1)
        overlap = None
        if previous_ids is not None:
            overlap = len(previous_ids & chosen_ids)
        steps.append(
            {
                "second_weight": second_weight,
                "mean_effective_percentile_by_anchor": [
                    float(value) for value in np.mean(effective, axis=0)
                ],
                "anchor_top100_coverage": branch_coverage(chosen, percentiles),
                "adjacent_top20_intersection": overlap,
                "track_ids": [int(library.track_ids[index]) for index in chosen],
            }
        )
        previous_ids = chosen_ids
    return {
        "case_id": case["id"],
        "operator": operator,
        "anchors": [case["first"], case["second"]],
        "steps": steps,
    }


def refine_record(
    library: queue_eval.Library,
    case: dict[str, object],
    by_track_id: dict[int, int],
    text_embeddings: dict[str, tuple[str, np.ndarray]],
) -> dict[str, object]:
    anchors = [
        resolve_anchor(
            (str(case["primary"]), 1.0),
            library,
            by_track_id,
            text_embeddings,
        ),
        resolve_anchor(
            (str(case["secondary"]), float(case["secondary_sign"])),
            library,
            by_track_id,
            text_embeddings,
        ),
    ]
    similarities, percentiles = per_anchor_data(library, anchors)
    excluded = {
        int(anchor.anchor_id.split(":", 1)[1])
        for anchor in anchors
        if anchor.anchor_id.startswith("song:")
    }
    primary_order = ranked_indices(
        similarities[0], library.track_ids, excluded, max(20, int(library.count * 0.05))
    )
    raw_top20 = primary_order[:20]
    thresholds: list[dict[str, object]] = []
    for fraction in REFINE_FRACTIONS:
        cutoff = 1.0 - fraction
        eligible = np.flatnonzero(percentiles[0] > cutoff)
        if excluded:
            eligible = eligible[
                ~np.isin(
                    library.track_ids[eligible],
                    np.fromiter(excluded, dtype=np.int64),
                )
            ]
        order = np.lexsort(
            (
                library.track_ids[eligible],
                -percentiles[0][eligible],
                -percentiles[1][eligible],
            )
        )
        chosen = eligible[order[:20]]
        thresholds.append(
            {
                "primary_top_fraction": fraction,
                "candidate_count": int(eligible.size),
                "mean_primary_percentile": float(np.mean(percentiles[0][chosen])),
                "min_primary_percentile": float(np.min(percentiles[0][chosen])),
                "mean_effective_secondary_percentile": float(
                    np.mean(percentiles[1][chosen])
                ),
                "min_effective_secondary_percentile": float(
                    np.min(percentiles[1][chosen])
                ),
                "overlap_with_raw_primary_top20": len(
                    set(chosen.tolist()) & set(raw_top20.tolist())
                ),
                "top_results": [
                    {
                        "rank": rank,
                        "track_id": int(library.track_ids[index]),
                        "artist": library.artists[index],
                        "title": library.titles[index],
                        "primary_percentile": float(percentiles[0][index]),
                        "effective_secondary_percentile": float(percentiles[1][index]),
                    }
                    for rank, index in enumerate(chosen, start=1)
                ],
            }
        )
    return {
        "case_id": case["id"],
        "anchors": anchor_records(anchors),
        "raw_primary_top20_mean_effective_secondary_percentile": float(
            np.mean(percentiles[1][raw_top20])
        ),
        "raw_primary_top20": [
            int(library.track_ids[index]) for index in raw_top20
        ],
        "thresholds": thresholds,
    }


def anchor_records(anchors: Sequence[Anchor]) -> list[dict[str, object]]:
    return [
        {"id": anchor.anchor_id, "label": anchor.label, "weight": anchor.weight}
        for anchor in anchors
    ]


def endpoint_checks(library: queue_eval.Library) -> dict[str, object]:
    anchor = Anchor("synthetic:a", "synthetic", library.embeddings[0], 1.0)
    sims, percentiles = per_anchor_data(library, [anchor])
    expected = ranked_indices(sims[0], library.track_ids, {int(library.track_ids[0])}, 50)
    checks: dict[str, bool] = {}
    for operator in ("balance", "direction", "either"):
        scores, error = operator_scores(operator, library, [anchor], percentiles)
        assert error is None and scores is not None
        actual = ranked_indices(scores, library.track_ids, {int(library.track_ids[0])}, 50)
        checks[f"single_anchor_{operator}"] = bool(np.array_equal(expected, actual))

    scaled = Anchor("synthetic:a", "synthetic", library.embeddings[0], 0.17)
    scaled_sims, scaled_percentiles = per_anchor_data(library, [scaled])
    scaled_scores, _ = operator_scores("direction", library, [scaled], scaled_percentiles)
    assert scaled_scores is not None
    scaled_order = ranked_indices(
        scaled_scores, library.track_ids, {int(library.track_ids[0])}, 50
    )
    checks["positive_weight_scaling_direction"] = bool(
        np.array_equal(expected, scaled_order)
    )

    contradiction = [anchor, Anchor("synthetic:a", "synthetic", anchor.embedding, -1.0)]
    contradiction_data = per_anchor_data(library, contradiction)[1]
    contradiction_scores, contradiction_error = operator_scores(
        "direction", library, contradiction, contradiction_data
    )
    checks["signed_copy_contradiction_rejected"] = (
        contradiction_scores is None and contradiction_error is not None
    )
    if not all(checks.values()):
        raise AssertionError(f"endpoint contract failed: {checks}")
    return checks


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_listening_packet(path: Path, records: Sequence[dict[str, object]]) -> None:
    lines = ["# Composition Listening Packet", ""]
    for record in records:
        lines.append(f"## {record['case_id']} / {record['operator']}")
        lines.append("")
        lines.append(
            "Anchors: "
            + "; ".join(
                f"{anchor['weight']:+g} {anchor['label']}" for anchor in record["anchors"]
            )
        )
        lines.append("")
        if record["status"] == "rejected":
            lines.append(f"Rejected: {record['reason']}")
            lines.append("")
            continue
        lines.append("| # | Track | Score | Effective anchor percentiles |")
        lines.append("|---:|---|---:|---|")
        for row in record["top_results"][:20]:
            track = f"{row['artist'] or '?'} - {row['title'] or '?'}".replace("|", "/")
            components = ", ".join(
                f"{value:.3f}" for value in row["effective_anchor_percentiles"]
            )
            lines.append(f"| {row['rank']} | {track} | {row['score']:.6f} | {components} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--text-results", type=Path, default=DEFAULT_TEXT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library, db_hash = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    text_embeddings = load_text_embeddings(args.text_results)
    by_track_id = track_index_by_id(library)
    records: list[dict[str, object]] = []
    for case_position, case in enumerate(PRIMARY_CASES, start=1):
        anchors = [
            resolve_anchor(spec, library, by_track_id, text_embeddings)
            for spec in case["anchors"]
        ]
        for operator in case["operators"]:
            record = result_record(library, str(case["id"]), str(operator), anchors)
            records.append(record)
            print(
                f"{case_position}/{len(PRIMARY_CASES)} {case['id']} / {operator}: "
                f"{record['status']} {record['elapsed_ms']:.1f} ms",
                flush=True,
            )

    weight_sweeps: list[dict[str, object]] = []
    for case in FOLLOWUP_WEIGHT_CASES:
        for operator in ("balance", "direction"):
            weight_sweeps.append(
                weight_sweep_record(
                    library,
                    case,
                    operator,
                    by_track_id,
                    text_embeddings,
                )
            )
            print(f"F01 {case['id']} / {operator}: complete", flush=True)

    refinements: list[dict[str, object]] = []
    for case in FOLLOWUP_REFINE_CASES:
        refinements.append(
            refine_record(library, case, by_track_id, text_embeddings)
        )
        print(f"F02 {case['id']}: complete", flush=True)

    payload = {
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "tracks": library.count,
            "dim": library.dim,
        },
        "text_results": {
            "path": str(args.text_results.resolve()),
            "sha256": sha256_file(args.text_results),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "top_k": TOP_K,
        "case_definition_sha256": hashlib.sha256(
            json.dumps(PRIMARY_CASES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "followup_weight_case_sha256": hashlib.sha256(
            json.dumps(FOLLOWUP_WEIGHT_CASES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "followup_refine_case_sha256": hashlib.sha256(
            json.dumps(FOLLOWUP_REFINE_CASES, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "endpoint_checks": endpoint_checks(library),
        "records": records,
        "followup_weight_sweeps": weight_sweeps,
        "followup_refinements": refinements,
    }
    atomic_json(args.output / "results.json", payload)
    write_listening_packet(args.output / "listening-packet.md", records)
    print(f"Complete. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
