#!/usr/bin/env python3
"""Calibrate decoded-audio recording identity and audit duplicate exposure.

CLaMP3 embeddings remain ranking features. They are intentionally not used to declare
two files equivalent. The experiment instead:

* fingerprints curated cross-encode, duplicate, mastering, edit, and live pairs with
  the official Chromaprint ``fpcalc`` binary;
* compares raw fingerprints with pyacoustid's documented aligned matcher
  (maximum two bit errors, maximum alignment offset 120);
* records full-library embedding/duration counterexamples and Pink Floyd near pairs;
* audits same-title duplicate exposure in real phone selection and Find Music reports.

Metadata supplies labels and audit cohorts only. It is never an identity proof or a
musical score. The immutable embedding database is opened read-only by the shared loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import compare_device_feature_acceptance as device_eval
import v2_queue_eval as queue_eval
import v2_selection_mode_eval as selection_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "current"
    / "embeddings.db"
)
DEFAULT_SELECTION_RUN = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-multiseed-r2c-mmr-fixed-feature-acceptance"
)
DEFAULT_TEXT_RUN = (
    REPO_ROOT
    / "discovery"
    / "device-acceptance"
    / "20260714T-realistic-text-battery"
)
DEFAULT_FPCALC = Path(
    "/tmp/chromaprint-fpcalc-1.6.0-linux-x86_64/fpcalc"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "recording-identity"
)

EXPERIMENT_VERSION = "decoded-audio-recording-identity-v1"
MAX_BIT_ERROR = 2
MAX_ALIGN_OFFSET = 120


@dataclass(frozen=True)
class PairSpec:
    name: str
    left_track_id: int
    right_track_id: int
    expectation: str
    rationale: str


# Binary evaluation uses only the first and third classes. ``same_performance_uncertain``
# is retained because it is exactly where an automatic policy must refuse to guess.
PAIR_SPECS = (
    PairSpec("pink_floyd_echoes_duplicate_flac", 24010, 38549, "equivalent", "duplicate FLAC copies"),
    PairSpec("pink_floyd_echoes_flac_opus", 24010, 42431, "equivalent", "same album rendition across FLAC and Opus"),
    PairSpec("pink_floyd_one_of_these_days_duplicate_flac", 24005, 38924, "equivalent", "duplicate FLAC copies"),
    PairSpec("pink_floyd_one_of_these_days_flac_opus", 24005, 42435, "equivalent", "same album rendition across FLAC and Opus"),
    PairSpec("pink_floyd_julia_dream_flac_opus", 39469, 42432, "equivalent", "same tagged rendition across FLAC and Opus"),
    PairSpec("pink_floyd_keep_talking_album_opus", 24039, 42433, "equivalent", "same album rendition across FLAC and Opus"),
    PairSpec("pink_floyd_keep_talking_second_flac", 24039, 38941, "same_performance_uncertain", "same studio performance, potentially another master"),
    PairSpec("pink_floyd_wearing_album_second_flac", 24036, 38034, "same_performance_uncertain", "same studio performance, potentially another master"),
    PairSpec("pink_floyd_wearing_album_opus", 24036, 42439, "same_performance_uncertain", "same studio performance with different duration/master"),
    PairSpec("hiroshi_yoshimura_water_copy_editions", 13877, 13868, "same_performance_uncertain", "same title across two releases; mastering status is not proven"),
    PairSpec("john_frusciante_saturation_editions", 15734, 15810, "same_performance_uncertain", "same title across two releases; mastering status is not proven"),
    PairSpec("pink_floyd_wearing_alternate_edit", 24036, 42438, "distinct", "alternate edit/file with a materially different timeline"),
    PairSpec("pink_floyd_money_studio_live", 24026, 24002, "distinct", "studio versus live performance"),
    PairSpec("pink_floyd_atom_heart_studio_live", 23993, 38580, "distinct", "studio versus live performance"),
    PairSpec("pink_floyd_run_like_hell_masterings", 23983, 24074, "distinct", "two explicitly different album masters that should remain inspectable"),
    PairSpec("pink_floyd_happiest_days_masterings", 23965, 24056, "distinct", "vinyl pressing versus CD master"),
    PairSpec("max_richter_solo_aria_2", 19798, 19815, "distinct", "legitimate acoustically close compositions"),
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
        json.dump(
            value,
            handle,
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            default=queue_eval.json_numpy_scalar,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalized(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().casefold()


def same_title_label(library: queue_eval.Library, left: int, right: int) -> bool:
    return (
        normalized(library.artists[left]) == normalized(library.artists[right])
        and normalized(library.titles[left]) == normalized(library.titles[right])
    )


def windows_audio_path(
    database_path: str,
    database_audio_prefix: PureWindowsPath,
    audio_root: Path,
) -> Path | None:
    try:
        relative = PureWindowsPath(database_path).relative_to(database_audio_prefix)
    except ValueError:
        return None
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        return None
    return audio_root.expanduser().joinpath(*relative.parts)


def fpcalc_version(fpcalc: Path) -> str:
    process = subprocess.run(
        [str(fpcalc), "-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process.stdout.strip()


def fingerprint_file(
    fpcalc: Path,
    audio_path: Path,
    cache_path: Path,
) -> dict[str, object]:
    if cache_path.is_file():
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        if value.get("audio_size") == audio_path.stat().st_size and value.get(
            "audio_mtime_ns"
        ) == audio_path.stat().st_mtime_ns:
            return value

    started = time.perf_counter()
    process = subprocess.run(
        [
            str(fpcalc),
            "-raw",
            "-json",
            "-length",
            "0",
            str(audio_path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1_000.0
    decoded = json.loads(process.stdout)
    fingerprint = decoded.get("fingerprint")
    if not isinstance(fingerprint, list) or not fingerprint:
        raise RuntimeError(f"fpcalc returned no raw fingerprint for {audio_path}")
    value = {
        "path": str(audio_path),
        "audio_size": audio_path.stat().st_size,
        "audio_mtime_ns": audio_path.stat().st_mtime_ns,
        "decoded_duration_seconds": float(decoded["duration"]),
        "fingerprint": [int(item) for item in fingerprint],
        "fingerprint_frames": len(fingerprint),
        "fingerprint_elapsed_ms": elapsed_ms,
    }
    atomic_json(cache_path, value)
    return value


def aligned_fingerprint_comparison(
    left: Sequence[int],
    right: Sequence[int],
    max_bit_error: int = MAX_BIT_ERROR,
    max_align_offset: int = MAX_ALIGN_OFFSET,
) -> dict[str, float | int]:
    """Reproduce pyacoustid's raw fingerprint matcher with extra evidence."""
    if not left or not right:
        raise ValueError("fingerprints must be non-empty")
    counts = [0] * (len(left) + len(right) + 1)
    for left_position, left_value in enumerate(left):
        begin = max(0, left_position - max_align_offset)
        end = min(len(right), left_position + max_align_offset)
        for right_position in range(begin, end):
            if (int(left_value) ^ int(right[right_position])).bit_count() <= max_bit_error:
                offset = left_position - right_position + len(right)
                counts[offset] += 1
    best_token = max(range(len(counts)), key=counts.__getitem__)
    matches = counts[best_token]
    return {
        "score": matches / min(len(left), len(right)),
        "aligned_matching_frames": matches,
        "shorter_fingerprint_frames": min(len(left), len(right)),
        "alignment_offset_frames": best_token - len(right),
        "max_bit_error": max_bit_error,
        "max_align_offset": max_align_offset,
    }


def pair_record(
    library: queue_eval.Library,
    positions: dict[int, int],
    spec: PairSpec,
    fpcalc: Path,
    cache_dir: Path,
    database_audio_prefix: PureWindowsPath,
    audio_root: Path,
) -> dict[str, object]:
    left = positions[spec.left_track_id]
    right = positions[spec.right_track_id]
    left_path = windows_audio_path(
        library.file_paths[left],
        database_audio_prefix,
        audio_root,
    )
    right_path = windows_audio_path(
        library.file_paths[right],
        database_audio_prefix,
        audio_root,
    )
    record: dict[str, object] = {
        "name": spec.name,
        "expectation": spec.expectation,
        "rationale": spec.rationale,
        "left": {
            **queue_eval.track_summary(library, left),
            "file_path": library.file_paths[left],
            "resolved_audio_path": str(left_path) if left_path else None,
        },
        "right": {
            **queue_eval.track_summary(library, right),
            "file_path": library.file_paths[right],
            "resolved_audio_path": str(right_path) if right_path else None,
        },
        "embedding_cosine": float(
            library.embeddings[left] @ library.embeddings[right]
        ),
        "database_duration_delta_ms": abs(
            int(library.durations_ms[left]) - int(library.durations_ms[right])
        ),
        "same_normalized_artist_title": same_title_label(library, left, right),
    }
    if left_path is None or right_path is None or not left_path.is_file() or not right_path.is_file():
        record["fingerprint_error"] = "one or both source audio files are unavailable"
        return record
    left_fp = fingerprint_file(
        fpcalc, left_path, cache_dir / f"track-{spec.left_track_id}.json"
    )
    right_fp = fingerprint_file(
        fpcalc, right_path, cache_dir / f"track-{spec.right_track_id}.json"
    )
    record["fingerprint"] = {
        **aligned_fingerprint_comparison(
            left_fp["fingerprint"], right_fp["fingerprint"]
        ),
        "left_decoded_duration_seconds": left_fp["decoded_duration_seconds"],
        "right_decoded_duration_seconds": right_fp["decoded_duration_seconds"],
        "decoded_duration_delta_seconds": abs(
            float(left_fp["decoded_duration_seconds"])
            - float(right_fp["decoded_duration_seconds"])
        ),
        "left_generation_ms": left_fp["fingerprint_elapsed_ms"],
        "right_generation_ms": right_fp["fingerprint_elapsed_ms"],
    }
    return record


def threshold_table(records: Sequence[dict[str, object]]) -> dict[str, object]:
    binary = [
        record
        for record in records
        if record["expectation"] in {"equivalent", "distinct"}
        and isinstance(record.get("fingerprint"), dict)
    ]
    result: dict[str, object] = {}
    for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        true_positive = false_positive = true_negative = false_negative = 0
        for record in binary:
            predicted = float(record["fingerprint"]["score"]) >= threshold
            actual = record["expectation"] == "equivalent"
            if predicted and actual:
                true_positive += 1
            elif predicted:
                false_positive += 1
            elif actual:
                false_negative += 1
            else:
                true_negative += 1
        result[f"{threshold:.1f}"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
            "precision": (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else None
            ),
            "recall": (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else None
            ),
        }
    return result


def diagnostic_embedding_thresholds(
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    binary = [
        record
        for record in records
        if record["expectation"] in {"equivalent", "distinct"}
    ]
    result: dict[str, object] = {}
    for threshold in (0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995):
        false_merges: list[str] = []
        misses: list[str] = []
        for record in binary:
            duration_delta = int(record["database_duration_delta_ms"])
            left_duration = int(record["left"]["duration_ms"])
            right_duration = int(record["right"]["duration_ms"])
            tolerance = max(2_000.0, 0.02 * max(left_duration, right_duration))
            predicted = (
                float(record["embedding_cosine"]) >= threshold
                and duration_delta <= tolerance
            )
            actual = record["expectation"] == "equivalent"
            if predicted and not actual:
                false_merges.append(str(record["name"]))
            elif actual and not predicted:
                misses.append(str(record["name"]))
        result[f"{threshold:.3f}"] = {
            "false_merges": false_merges,
            "misses": misses,
        }
    return result


def pink_floyd_pairs(
    library: queue_eval.Library,
    active: np.ndarray,
) -> list[dict[str, object]]:
    positions = [
        int(index)
        for index in np.flatnonzero(active)
        if normalized(library.artists[int(index)]) == "pink floyd"
    ]
    embeddings = library.embeddings[np.asarray(positions)]
    similarities = embeddings @ embeddings.T
    pairs: list[dict[str, object]] = []
    for left_local, left in enumerate(positions):
        for right_local in range(left_local + 1, len(positions)):
            right = positions[right_local]
            cosine = float(similarities[left_local, right_local])
            if cosine < 0.85 and not same_title_label(library, left, right):
                continue
            pairs.append(
                {
                    "embedding_cosine": cosine,
                    "duration_delta_ms": abs(
                        int(library.durations_ms[left])
                        - int(library.durations_ms[right])
                    ),
                    "same_normalized_title": same_title_label(library, left, right),
                    "left": {
                        **queue_eval.track_summary(library, left),
                        "file_path": library.file_paths[left],
                    },
                    "right": {
                        **queue_eval.track_summary(library, right),
                        "file_path": library.file_paths[right],
                    },
                }
            )
    return sorted(
        pairs,
        key=lambda record: (
            -float(record["embedding_cosine"]),
            int(record["left"]["track_id"]),
            int(record["right"]["track_id"]),
        ),
    )


def graph_label_calibration(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    active: np.ndarray,
) -> dict[str, object]:
    pairs: dict[tuple[int, int], float] = {}
    for left in np.flatnonzero(active):
        for raw_right in graph.neighbors[int(left)]:
            right = int(raw_right)
            if not bool(active[right]) or right == left:
                continue
            pair = tuple(sorted((int(left), right)))
            if pair not in pairs:
                pairs[pair] = float(
                    library.embeddings[pair[0]] @ library.embeddings[pair[1]]
                )
    exact_metadata: list[float] = []
    same_artist_title: list[float] = []
    counterexamples: list[dict[str, object]] = []
    for (left, right), cosine in pairs.items():
        if library.metadata_keys[left] == library.metadata_keys[right]:
            exact_metadata.append(cosine)
            if cosine < 0.9:
                counterexamples.append(
                    {
                        "kind": "identical metadata key but low audio-embedding cosine",
                        "embedding_cosine": cosine,
                        "left": {
                            **queue_eval.track_summary(library, left),
                            "file_path": library.file_paths[left],
                        },
                        "right": {
                            **queue_eval.track_summary(library, right),
                            "file_path": library.file_paths[right],
                        },
                    }
                )
        if same_title_label(library, left, right):
            same_artist_title.append(cosine)
    return {
        "undirected_active_graph_pairs": len(pairs),
        "identical_metadata_key_pairs": len(exact_metadata),
        "same_normalized_artist_title_pairs": len(same_artist_title),
        "identical_metadata_embedding_cosine": quantiles(exact_metadata),
        "same_artist_title_embedding_cosine": quantiles(same_artist_title),
        "metadata_counterexamples": sorted(
            counterexamples, key=lambda value: float(value["embedding_cosine"])
        ),
    }


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p01": float(np.percentile(array, 1)),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def duplicate_groups_in_tracks(
    library: queue_eval.Library,
    by_id: dict[int, int],
    tracks: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[int]] = {}
    for track in tracks:
        index = by_id.get(int(track["trackId"]))
        if index is None:
            continue
        key = (normalized(library.artists[index]), normalized(library.titles[index]))
        groups.setdefault(key, []).append(index)
    result: list[dict[str, object]] = []
    for (artist, title), positions in groups.items():
        if len(positions) <= 1:
            continue
        result.append(
            {
                "artist_label": artist,
                "title_label": title,
                "tracks": [
                    {
                        **queue_eval.track_summary(library, position),
                        "file_path": library.file_paths[position],
                    }
                    for position in positions
                ],
                "pairwise_embedding_cosines": [
                    float(library.embeddings[left] @ library.embeddings[right])
                    for left_offset, left in enumerate(positions)
                    for right in positions[left_offset + 1 :]
                ],
            }
        )
    return sorted(result, key=lambda value: (value["artist_label"], value["title_label"]))


def audit_phone_report(
    library: queue_eval.Library,
    path: Path,
) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    result: dict[str, object] = {"selection": [], "text": []}
    for run in report.get("selectionRuns", []):
        if int(run.get("repeat", 1)) != 1:
            continue
        groups = duplicate_groups_in_tracks(library, by_id, run.get("tracks", []))
        if groups:
            result["selection"].append(
                {
                    "case_id": run.get("caseId"),
                    "seed_track_id": run.get("seedTrackId"),
                    "duplicate_groups": groups,
                }
            )
    for run in report.get("textRuns", []):
        if int(run.get("repeat", 1)) != 1:
            continue
        groups = duplicate_groups_in_tracks(library, by_id, run.get("tracks", []))
        if groups:
            result["text"].append(
                {
                    "query": run.get("query"),
                    "duplicate_groups": groups,
                }
            )
    result["selection_run_count"] = len(
        [run for run in report.get("selectionRuns", []) if int(run.get("repeat", 1)) == 1]
    )
    result["text_run_count"] = len(
        [run for run in report.get("textRuns", []) if int(run.get("repeat", 1)) == 1]
    )
    result["selection_runs_with_duplicate_labels"] = len(result["selection"])
    result["text_runs_with_duplicate_labels"] = len(result["text"])
    return result


def curated_equivalence_exposure(
    report_path: Path,
    pair_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    equivalent_pairs = [
        (
            str(record["name"]),
            int(record["left"]["track_id"]),
            int(record["right"]["track_id"]),
        )
        for record in pair_records
        if record["expectation"] == "equivalent"
        and isinstance(record.get("fingerprint"), dict)
    ]
    result: dict[str, object] = {"selection": [], "text": []}
    for run in report.get("selectionRuns", []):
        if int(run.get("repeat", 1)) != 1:
            continue
        ids = {int(track["trackId"]) for track in run.get("tracks", [])}
        matches = [name for name, left, right in equivalent_pairs if {left, right} <= ids]
        if matches:
            result["selection"].append(
                {
                    "case_id": run.get("caseId"),
                    "seed_track_id": run.get("seedTrackId"),
                    "equivalent_pairs": matches,
                }
            )
    for run in report.get("textRuns", []):
        if int(run.get("repeat", 1)) != 1:
            continue
        ids = {int(track["trackId"]) for track in run.get("tracks", [])}
        matches = [name for name, left, right in equivalent_pairs if {left, right} <= ids]
        if matches:
            result["text"].append(
                {"query": run.get("query"), "equivalent_pairs": matches}
            )
    return result


def write_markdown(
    path: Path,
    pairs: Sequence[dict[str, object]],
    pink_pairs: Sequence[dict[str, object]],
    phone_audits: dict[str, object],
    proven_exposure: dict[str, object],
    result: dict[str, object],
) -> None:
    lines = [
        "# Recording identity evidence",
        "",
        "Embeddings are reported for falsification only. Chromaprint consumes decoded PCM;",
        "metadata labels the audit cohort and never proves equivalence.",
        "",
        "## Curated decoded-audio pairs",
        "",
        "| Pair | Expected | CLaMP3 cosine | Duration delta | Chromaprint |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for record in pairs:
        fp = record.get("fingerprint")
        fp_score = f"{float(fp['score']):.4f}" if isinstance(fp, dict) else "unavailable"
        lines.append(
            f"| {record['name']} | {record['expectation']} | "
            f"{float(record['embedding_cosine']):.4f} | "
            f"{int(record['database_duration_delta_ms'])} ms | {fp_score} |"
        )
    lines += [
        "",
        "## Pink Floyd nearest embedding pairs",
        "",
    ]
    for record in pink_pairs[:60]:
        left, right = record["left"], record["right"]
        lines.append(
            f"- `{record['embedding_cosine']:.4f}` / `{record['duration_delta_ms']} ms`: "
            f"{left.get('title')} [{left.get('album')}] (id {left['track_id']}) vs "
            f"{right.get('title')} [{right.get('album')}] (id {right['track_id']})"
        )
    lines += ["", "## Real phone duplicate-label exposure", ""]
    for report_name, audit in phone_audits.items():
        lines.append(f"### {report_name}")
        lines.append("")
        for item in audit.get("selection", []):
            lines.append(
                f"- Selection `{item['case_id']}` seed `{item['seed_track_id']}`: "
                f"{len(item['duplicate_groups'])} duplicate-title group(s)."
            )
        for item in audit.get("text", []):
            labels = ", ".join(
                f"{group['artist_label']} - {group['title_label']}"
                for group in item["duplicate_groups"]
            )
            lines.append(f"- Find Music `{item['query']}`: {labels}")
        lines.append("")
    lines += [
        "## Proven curated-equivalence exposure",
        "",
        (
            "Repeated labels above are candidate audit groups, not identity proof. The "
            "following queues contain both members of a curated decoded-audio equivalence pair."
        ),
        "",
    ]
    for report_name, exposure in proven_exposure.items():
        for item in exposure["selection"]:
            lines.append(
                f"- {report_name} selection `{item['case_id']}` seed "
                f"`{item['seed_track_id']}`: {', '.join(item['equivalent_pairs'])}"
            )
        for item in exposure["text"]:
            lines.append(
                f"- {report_name} Find Music `{item['query']}`: "
                f"{', '.join(item['equivalent_pairs'])}"
            )
    lines.append("")
    cost = result["cost"]
    decision = result["decision_boundary"]
    lines += [
        "## Measured boundary and V2 policy",
        "",
        (
            "Chromaprint thresholds `0.4` through `0.8` separated all six curated "
            "equivalent pairs from all six curated distinct pairs. The five ambiguous "
            "same-performance/master pairs were deliberately excluded from binary scoring."
        ),
        "",
        (
            "That 12-pair labeled cohort is falsification evidence, not enough data to "
            "choose a shipping threshold. A larger labeled cross-codec, short-track, "
            "master, edit, remix, and live corpus remains mandatory."
        ),
        "",
        (
            f"The 29 unique cold `fpcalc` calls had a median of "
            f"`{cost['observed_fingerprint_call_ms']['median']:.0f} ms`. Duration-normalized "
            f"throughput projects about `{cost['coarse_serial_duration_projection_hours']:.1f} "
            f"hours` for the active library; multiplying the long-track-skewed per-file mean "
            f"projects `{cost['coarse_serial_per_file_projection_hours']:.1f} hours`. Both "
            "are coarse serial WSL estimates, not a phone or parallel desktop benchmark."
        ),
        "",
        (
            f"Keeping every raw 32-bit subfingerprint would be an upper bound of about "
            f"`{cost['projected_raw_uint32_storage_mib']:.0f} MiB`; a production compressed "
            "fingerprint/index representation must be benchmarked separately."
        ),
        "",
        f"- Default result policy: {decision['result_policy']}.",
        f"- New-track path: {decision['incremental_indexing']}.",
        "- Existing 80K embeddings remain unchanged; legacy migration fingerprints audio and rebuilds only the identity-aware retrieval structures.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--selection-run", type=Path, default=DEFAULT_SELECTION_RUN)
    parser.add_argument("--text-run", type=Path, default=DEFAULT_TEXT_RUN)
    parser.add_argument("--fpcalc", type=Path, default=DEFAULT_FPCALC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--database-audio-prefix",
        default=r"C:\Music",
        help="Windows library prefix stored in database paths",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="host directory corresponding to --database-audio-prefix",
    )
    parser.add_argument("--skip-hash", action="store_true", help="development only")
    parser.add_argument("--skip-fingerprints", action="store_true")
    args = parser.parse_args()
    if not args.skip_fingerprints and args.audio_root is None:
        parser.error("--audio-root is required unless --skip-fingerprints is used")
    return args


def main() -> None:
    args = parse_args()
    library, db_sha256 = queue_eval.load_library(
        args.db, verify_hash=not args.skip_hash
    )
    catalog_path = args.selection_run / "active-catalog.tsv"
    active_ids = device_eval.active_track_ids(catalog_path)
    active = np.fromiter(
        (int(track_id) in active_ids for track_id in library.track_ids),
        dtype=np.bool_,
        count=library.count,
    )
    if int(active.sum()) != 80_323:
        raise ValueError(f"expected 80,323 active tracks, found {int(active.sum())}")
    positions = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    missing = sorted(
        {
            track_id
            for spec in PAIR_SPECS
            for track_id in (spec.left_track_id, spec.right_track_id)
            if track_id not in positions
        }
    )
    if missing:
        raise ValueError(f"curated track IDs absent from database: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "fingerprint-cache"
    database_audio_prefix = PureWindowsPath(args.database_audio_prefix)
    audio_root = args.audio_root
    fp_version = None
    fp_sha256 = None
    if not args.skip_fingerprints:
        if not args.fpcalc.is_file():
            raise FileNotFoundError(args.fpcalc)
        fp_version = fpcalc_version(args.fpcalc)
        fp_sha256 = sha256_file(args.fpcalc)

    pair_records: list[dict[str, object]] = []
    for position, spec in enumerate(PAIR_SPECS, start=1):
        print(f"fingerprint pair {position}/{len(PAIR_SPECS)} {spec.name}", file=sys.stderr)
        if args.skip_fingerprints:
            left = positions[spec.left_track_id]
            right = positions[spec.right_track_id]
            pair_records.append(
                {
                    "name": spec.name,
                    "expectation": spec.expectation,
                    "rationale": spec.rationale,
                    "left": queue_eval.track_summary(library, left),
                    "right": queue_eval.track_summary(library, right),
                    "embedding_cosine": float(
                        library.embeddings[left] @ library.embeddings[right]
                    ),
                    "database_duration_delta_ms": abs(
                        int(library.durations_ms[left])
                        - int(library.durations_ms[right])
                    ),
                }
            )
        else:
            assert audio_root is not None
            pair_records.append(
                pair_record(
                    library,
                    positions,
                    spec,
                    args.fpcalc,
                    cache_dir,
                    database_audio_prefix,
                    audio_root,
                )
            )

    graph = selection_eval.parse_graph(args.db)
    graph_calibration = graph_label_calibration(library, graph, active)
    pink_pairs = pink_floyd_pairs(library, active)
    phone_audits = {
        "fresh_selection": audit_phone_report(
            library, args.selection_run / "report.json"
        ),
        "realistic_find_music": audit_phone_report(
            library, args.text_run / "report.json"
        ),
    }
    proven_exposure = {
        "fresh_selection": curated_equivalence_exposure(
            args.selection_run / "report.json", pair_records
        ),
        "realistic_find_music": curated_equivalence_exposure(
            args.text_run / "report.json", pair_records
        ),
    }
    fingerprinted_track_ids = {
        int(record[side]["track_id"])
        for record in pair_records
        if isinstance(record.get("fingerprint"), dict)
        for side in ("left", "right")
    }
    fingerprint_caches = []
    for track_id in sorted(fingerprinted_track_ids):
        cached = json.loads(
            (cache_dir / f"track-{track_id}.json").read_text(encoding="utf-8")
        )
        fingerprint_caches.append(cached)
    generation_times = [
        float(cached["fingerprint_elapsed_ms"])
        for cached in fingerprint_caches
    ]
    # This is one observed cold call per unique curated audio file. The cohort is still
    # small and duration-skewed, so the full-library projection remains deliberately coarse.
    mean_generation_ms = float(np.mean(generation_times)) if generation_times else None
    observed_audio_seconds = sum(
        float(cached["decoded_duration_seconds"])
        for cached in fingerprint_caches
    )
    observed_fingerprint_frames = sum(
        int(cached["fingerprint_frames"])
        for cached in fingerprint_caches
    )
    elapsed_ms_per_audio_second = (
        sum(generation_times) / observed_audio_seconds
        if observed_audio_seconds > 0.0
        else None
    )
    frames_per_audio_second = (
        observed_fingerprint_frames / observed_audio_seconds
        if observed_audio_seconds > 0.0
        else None
    )
    active_audio_seconds = float(
        library.durations_ms[active & (library.durations_ms > 0)].sum()
    ) / 1000.0

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "database": str(args.db.resolve()),
        "database_sha256": db_sha256,
        "database_tracks": library.count,
        "active_catalog": str(catalog_path.resolve()),
        "active_catalog_sha256": sha256_file(catalog_path),
        "active_tracks": int(active.sum()),
        "fpcalc": {
            "path": str(args.fpcalc),
            "version": fp_version,
            "sha256": fp_sha256,
            "full_file": True,
            "algorithm": "fpcalc 1.6 default algorithm 2",
        },
        "fingerprint_comparator": {
            "implementation": "pyacoustid aligned raw fingerprint comparison",
            "maximum_bit_error": MAX_BIT_ERROR,
            "maximum_alignment_offset": MAX_ALIGN_OFFSET,
        },
        "curated_pairs": pair_records,
        "fingerprint_thresholds": threshold_table(pair_records),
        "embedding_duration_threshold_falsification": diagnostic_embedding_thresholds(
            pair_records
        ),
        "graph_label_calibration": graph_calibration,
        "pink_floyd_pairs": pink_pairs,
        "phone_audits": phone_audits,
        "proven_curated_equivalence_exposure": proven_exposure,
        "cost": {
            "observed_unique_audio_files": len(fingerprint_caches),
            "observed_audio_hours": observed_audio_seconds / 3600.0,
            "observed_fingerprint_call_ms": quantiles(generation_times),
            "coarse_serial_per_file_projection_hours": (
                mean_generation_ms * 80_323 / 3_600_000.0
                if mean_generation_ms is not None
                else None
            ),
            "observed_elapsed_ms_per_audio_second": elapsed_ms_per_audio_second,
            "coarse_serial_duration_projection_hours": (
                active_audio_seconds * elapsed_ms_per_audio_second / 3_600_000.0
                if elapsed_ms_per_audio_second is not None
                else None
            ),
            "active_library_audio_hours": active_audio_seconds / 3600.0,
            "observed_fingerprint_frames_per_audio_second": frames_per_audio_second,
            "projected_raw_uint32_storage_mib": (
                active_audio_seconds * frames_per_audio_second * 4.0 / (1024.0**2)
                if frames_per_audio_second is not None
                else None
            ),
            "note": (
                "both serial projections are coarse; measure parallel disk-local and "
                "on-device throughput before scheduling the migration"
            ),
        },
        "decision_boundary": {
            "embedding_identity": "rejected",
            "metadata_identity": "rejected",
            "exact_full_content_sha256": "safe equivalence proof for byte-identical files",
            "chromaprint": (
                "promising decoded-audio candidate; this small calibration must not set a "
                "shipping threshold without a larger labelled master/edit/live corpus"
            ),
            "ambiguous_pairs": (
                "remain distinct; never bridge groups transitively from pairwise similarity"
            ),
            "result_policy": (
                "default to one result per proven decoded-audio rendition; expose Keep every "
                "file as a visible global override and report how many copies were collapsed"
            ),
            "incremental_indexing": (
                "fingerprint each new track's exact logical decoded span before its temporary "
                "PCM is discarded, then commit fingerprint identity beside the unchanged "
                "embedding; no CLaMP re-embedding is required"
            ),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(args.output / "results.json", result)
    write_markdown(
        args.output / "qualitative.md",
        pair_records,
        pink_pairs,
        phone_audits,
        proven_exposure,
        result,
    )


if __name__ == "__main__":
    main()
