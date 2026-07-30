#!/usr/bin/env python3
"""Falsify conservative, audio-only same-rendition grouping policies.

This experiment deliberately separates coarse recording recognition from proof that two
files are the same rendition. It combines:

* byte-identical duplicates from the real library;
* explicit edit, remix, live, and mastering negatives from the real library;
* controlled MP3/Opus transcodes, trims, and mastering transforms generated from three
  source tracks; and
* ambiguous real cross-encodes, which are reported but never used as binary truth.

Metadata is used only to locate and describe the curated audit files. It is never an
identity feature. CLaMP embeddings are not loaded or consulted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import correlate, correlation_lags


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_recording_identity_eval as identity_v1


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "2026-07-07T223308+0300_qv7706c3mq"
    / "embeddings.db"
)
DEFAULT_FPCALC = Path("/tmp/chromaprint-fpcalc-1.6.0-linux-x86_64/fpcalc")
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "recording-identity-policy-2026-07-15"
)
EXPERIMENT_VERSION = "audio-rendition-proof-falsification-v1"
PCM_SAMPLE_RATE = 8_000


@dataclass(frozen=True)
class RealPair:
    name: str
    left_track_id: int
    right_track_id: int
    expectation: str
    subtype: str
    rationale: str


@dataclass(frozen=True)
class ControlledSource:
    track_id: int
    description: str


@dataclass(frozen=True)
class Transform:
    name: str
    extension: str
    ffmpeg_args: tuple[str, ...]
    expectation: str
    subtype: str


REAL_PAIRS = (
    RealPair("echoes_duplicate_flac", 24010, 38549, "same_rendition", "exact_duplicate", "duplicate FLAC files"),
    RealPair("one_of_these_days_duplicate_flac", 24005, 38924, "same_rendition", "exact_duplicate", "duplicate FLAC files"),
    RealPair("deadcrush_ben_remix_duplicate_flac", 1983, 40344, "same_rendition", "exact_duplicate", "duplicate FLAC files"),
    RealPair("land_of_gold_original_radio_edit", 2514, 41392, "distinct", "edit", "album version versus radio edit"),
    RealPair("land_of_gold_mogwai_remix_edit", 2503, 2504, "distinct", "edit", "full remix versus named remix edit"),
    RealPair("land_of_gold_original_mogwai_remix", 2514, 2503, "distinct", "remix", "original versus named remix"),
    RealPair("deadcrush_original_ben_remix", 46247, 1983, "distinct", "remix", "original versus named remix"),
    RealPair("comfy_in_nautica_original_remix", 332, 339, "distinct", "remix", "original versus named remix"),
    RealPair("money_studio_live", 24026, 24002, "distinct", "live", "studio versus live performance"),
    RealPair("atom_heart_mother_studio_live", 23993, 38580, "distinct", "live", "studio versus live performance"),
    RealPair("run_like_hell_distinct_masters", 23983, 24074, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("happiest_days_distinct_masters", 23965, 24056, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("goodbye_blue_sky_distinct_masters", 23968, 24059, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("in_the_flesh_distinct_masters", 23982, 24073, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("hey_you_distinct_masters", 23975, 24066, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("waiting_for_the_worms_distinct_masters", 23984, 24075, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("show_must_go_on_distinct_masters", 23981, 24072, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("is_there_anybody_out_there_distinct_masters", 23976, 24067, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("another_brick_part_2_distinct_masters", 23966, 24057, "distinct", "master", "vinyl pressing transfer versus CD master"),
    RealPair("dont_leave_me_now_distinct_masters", 23972, 24063, "distinct", "master", "vinyl pressing transfer versus CD master"),
    # These identify the same musical recording, but the existing evidence does not prove
    # that the encoded files came from the same master. They must not train a threshold.
    RealPair("echoes_flac_opus_unproven_master", 24010, 42431, "ambiguous", "cross_encode", "same title/release label; source master is unproven"),
    RealPair("one_of_these_days_flac_opus_unproven_master", 24005, 42435, "ambiguous", "cross_encode", "same title/release label; source master is unproven"),
    RealPair("julia_dream_flac_opus_unproven_master", 39469, 42432, "ambiguous", "cross_encode", "same title/release label; source master is unproven"),
    RealPair("keep_talking_flac_opus_unproven_master", 24039, 42433, "ambiguous", "cross_encode", "same title/release label; source master is unproven"),
)


CONTROLLED_SOURCES = (
    ControlledSource(39469, "short rock song"),
    ControlledSource(1983, "dense electronic remix"),
    ControlledSource(13877, "quiet ambient recording"),
)


TRANSFORMS = (
    Transform("opus_64k", "opus", ("-c:a", "libopus", "-b:a", "64k"), "same_rendition", "controlled_transcode"),
    Transform("mp3_128k", "mp3", ("-c:a", "libmp3lame", "-b:a", "128k"), "same_rendition", "controlled_transcode"),
    Transform("trim_start_5s", "flac", ("-ss", "5", "-c:a", "flac"), "distinct", "controlled_edit"),
    Transform("trim_end_5s", "flac", ("-af", "areverse,atrim=start=5,areverse", "-c:a", "flac"), "distinct", "controlled_edit"),
    Transform("gain_plus_1db", "flac", ("-af", "volume=1dB", "-c:a", "flac"), "distinct", "controlled_master"),
    Transform("bass_plus_1db", "flac", ("-af", "bass=g=1:f=150:w=0.7", "-c:a", "flac"), "distinct", "controlled_master"),
    Transform(
        "moderate_compression",
        "flac",
        ("-af", "acompressor=threshold=0.125:ratio=2:attack=20:release=250:makeup=1", "-c:a", "flac"),
        "distinct",
        "controlled_master",
    ),
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def track_path(connection: sqlite3.Connection, track_id: int) -> tuple[str, Path]:
    row = connection.execute("SELECT file_path FROM tracks WHERE id = ?", (track_id,)).fetchone()
    if row is None:
        raise ValueError(f"missing track id {track_id}")
    database_path = str(row[0])
    resolved = identity_v1.windows_audio_path(database_path)
    if resolved is None or not resolved.is_file():
        raise FileNotFoundError(f"unavailable audio for track {track_id}: {database_path}")
    return database_path, resolved


def run_ffmpeg_atomic(source: Path, target: Path, args: Sequence[str]) -> None:
    if target.is_file() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(source), "-vn", *args, "-y", str(temporary)],
        check=True,
    )
    temporary.replace(target)


def decode_pcm(source: Path, cache_path: Path) -> np.memmap:
    if not cache_path.is_file() or not cache_path.stat().st_size:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.f32")
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(source),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(PCM_SAMPLE_RATE),
                "-f",
                "f32le",
                "-y",
                str(temporary),
            ],
            check=True,
        )
        temporary.replace(cache_path)
    return np.memmap(cache_path, dtype="<f4", mode="r")


def canonical_pcm_identity(source: Path, cache_path: Path) -> dict[str, object]:
    """Hash full-resolution decoded samples without resampling or downmixing."""
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("audio_size") == source.stat().st_size
            and cached.get("audio_mtime_ns") == source.stat().st_mtime_ns
        ):
            return cached
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels,channel_layout",
            "-of",
            "json",
            str(source),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    decoded_hash = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_f64le",
            "-f",
            "hash",
            "-hash",
            "sha256",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if not decoded_hash.startswith("SHA256="):
        raise RuntimeError(f"unexpected ffmpeg hash output for {source}: {decoded_hash}")
    value = {
        "spec": "ffmpeg-native-rate-and-channel-layout-pcm_f64le-sha256-v1",
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "channel_layout": stream.get("channel_layout"),
        "sha256": decoded_hash.removeprefix("SHA256="),
        "audio_size": source.stat().st_size,
        "audio_mtime_ns": source.stat().st_mtime_ns,
    }
    atomic_json(cache_path, value)
    return value


def align_pcm(left: np.ndarray, right: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    search_samples = min(len(left), len(right), 120 * PCM_SAMPLE_RATE)
    left_search = np.asarray(left[:search_samples], dtype=np.float64)
    right_search = np.asarray(right[:search_samples], dtype=np.float64)
    left_search -= left_search.mean()
    right_search -= right_search.mean()
    values = correlate(right_search, left_search, mode="full", method="fft")
    lags = correlation_lags(len(right_search), len(left_search), mode="full")
    eligible = np.abs(lags) <= 15 * PCM_SAMPLE_RATE
    eligible_values = np.abs(values[eligible])
    lag = int(lags[eligible][int(np.argmax(eligible_values))])
    if lag >= 0:
        count = min(len(left), len(right) - lag)
        aligned_left = np.asarray(left[:count], dtype=np.float64)
        aligned_right = np.asarray(right[lag : lag + count], dtype=np.float64)
    else:
        count = min(len(right), len(left) + lag)
        aligned_right = np.asarray(right[:count], dtype=np.float64)
        aligned_left = np.asarray(left[-lag : -lag + count], dtype=np.float64)
    return lag, aligned_left, aligned_right


def pcm_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float | int]:
    original_left_count = len(left)
    original_right_count = len(right)
    lag, aligned_left, aligned_right = align_pcm(left, right)
    centered_left = aligned_left - aligned_left.mean()
    centered_right = aligned_right - aligned_right.mean()
    correlation = abs(
        float(
            np.dot(centered_left, centered_right)
            / (np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
        )
    )

    block_size = PCM_SAMPLE_RATE
    complete_count = (len(aligned_left) // block_size) * block_size
    left_blocks = aligned_left[:complete_count].reshape(-1, block_size)
    right_blocks = aligned_right[:complete_count].reshape(-1, block_size)
    left_rms = np.sqrt(np.mean(left_blocks**2, axis=1) + 1e-12)
    right_rms = np.sqrt(np.mean(right_blocks**2, axis=1) + 1e-12)
    active = (left_rms > 1e-4) & (right_rms > 1e-4)
    block_gain = 20 * np.log10(right_rms[active] / left_rms[active])
    median_gain = float(np.median(block_gain))
    dynamic_delta = float(np.percentile(np.abs(block_gain - median_gain), 95))

    return {
        "sample_rate": PCM_SAMPLE_RATE,
        "alignment_lag_samples": lag,
        "alignment_lag_ms": lag * 1_000.0 / PCM_SAMPLE_RATE,
        "decoded_length_ratio": min(original_left_count, original_right_count)
        / max(original_left_count, original_right_count),
        "aligned_coverage_of_longer": len(aligned_left)
        / max(original_left_count, original_right_count),
        "waveform_absolute_correlation": correlation,
        "median_gain_delta_db": median_gain,
        "dynamic_delta_p95_db": dynamic_delta,
    }


def compare_pair(
    fpcalc: Path,
    left_path: Path,
    right_path: Path,
    left_key: str,
    right_key: str,
    output: Path,
    include_pcm: bool,
) -> dict[str, object]:
    decoded_identity_cache = output / "decoded-identity-cache"
    left_identity = canonical_pcm_identity(
        left_path, decoded_identity_cache / f"{left_key}.json"
    )
    right_identity = canonical_pcm_identity(
        right_path, decoded_identity_cache / f"{right_key}.json"
    )
    fingerprints = output / "fingerprint-cache"
    left_fp = identity_v1.fingerprint_file(fpcalc, left_path, fingerprints / f"{left_key}.json")
    right_fp = identity_v1.fingerprint_file(fpcalc, right_path, fingerprints / f"{right_key}.json")
    comparison = identity_v1.aligned_fingerprint_comparison(
        left_fp["fingerprint"], right_fp["fingerprint"]
    )
    comparison["mutual_coverage"] = comparison["aligned_matching_frames"] / max(
        len(left_fp["fingerprint"]), len(right_fp["fingerprint"])
    )
    result: dict[str, object] = {
        "byte_identical": (
            left_path.stat().st_size == right_path.stat().st_size
            and sha256_file(left_path) == sha256_file(right_path)
        ),
        "decoded_duration_delta_seconds": abs(
            float(left_fp["decoded_duration_seconds"])
            - float(right_fp["decoded_duration_seconds"])
        ),
        "decoded_pcm_identical": all(
            left_identity[key] == right_identity[key]
            for key in ("spec", "sample_rate", "channels", "channel_layout", "sha256")
        ),
        "left_decoded_pcm_identity": left_identity,
        "right_decoded_pcm_identity": right_identity,
        "chromaprint": comparison,
    }
    if include_pcm or float(comparison["score"]) >= 0.8:
        pcm_cache = output / "pcm-cache"
        left_pcm = decode_pcm(left_path, pcm_cache / f"{left_key}.f32")
        right_pcm = decode_pcm(right_path, pcm_cache / f"{right_key}.f32")
        result["pcm"] = pcm_metrics(left_pcm, right_pcm)
    return result


def predicts_same(record: dict[str, object], policy: str) -> bool:
    evidence = record["evidence"]
    if policy == "byte_identical":
        return bool(evidence["byte_identical"])
    if policy == "decoded_pcm_identical":
        return bool(evidence.get("decoded_pcm_identical"))
    if "chromaprint" not in evidence:
        return False
    chromaprint = evidence["chromaprint"]
    pcm = evidence.get("pcm")
    if policy == "chromaprint_0_8":
        return float(chromaprint["score"]) >= 0.8
    if policy == "chromaprint_0_8_mutual_0_99":
        return (
            float(chromaprint["score"]) >= 0.8
            and float(chromaprint["mutual_coverage"]) >= 0.99
        )
    if policy == "post_hoc_composite_not_validated":
        return (
            isinstance(pcm, dict)
            and float(chromaprint["score"]) >= 0.989
            and float(evidence["decoded_duration_delta_seconds"]) <= 3.0
            and float(pcm["waveform_absolute_correlation"]) >= 0.994
            and abs(float(pcm["median_gain_delta_db"])) <= 0.5
        )
    raise ValueError(policy)


def policy_table(records: Sequence[dict[str, object]]) -> dict[str, object]:
    binary = [
        record
        for record in records
        if record["expectation"] != "ambiguous"
        and "chromaprint" in record["evidence"]
    ]
    result: dict[str, object] = {}
    for policy in (
        "byte_identical",
        "decoded_pcm_identical",
        "chromaprint_0_8",
        "chromaprint_0_8_mutual_0_99",
        "post_hoc_composite_not_validated",
    ):
        false_merges: list[str] = []
        misses: list[str] = []
        true_positive = true_negative = 0
        for record in binary:
            predicted = predicts_same(record, policy)
            actual = record["expectation"] == "same_rendition"
            if predicted and actual:
                true_positive += 1
            elif predicted:
                false_merges.append(str(record["name"]))
            elif actual:
                misses.append(str(record["name"]))
            else:
                true_negative += 1
        result[policy] = {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_merges": false_merges,
            "misses": misses,
        }
    return result


def write_summary(path: Path, result: dict[str, object]) -> None:
    records = result["records"]
    policies = result["policies"]
    lines = [
        "# Audio rendition proof falsification",
        "",
        "This calibration uses audio only for every decision. Metadata only locates the",
        "curated files; ambiguous real cross-encodes do not count as binary truth.",
        "",
        "## Cohort",
        "",
        f"- Binary same-rendition cases: `{sum(r['expectation'] == 'same_rendition' for r in records)}`.",
        f"- Binary distinct edit/remix/live/master cases: `{sum(r['expectation'] == 'distinct' for r in records)}`.",
        f"- Unscored ambiguous cross-encodes: `{sum(r['expectation'] == 'ambiguous' for r in records)}`.",
        f"- Decoder failures excluded from policy scoring: `{sum('chromaprint' not in r['evidence'] for r in records)}`.",
        "",
        "## Policy falsification",
        "",
        "| Policy | Same-rendition hits | Distinct preserved | False merges | Misses |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, value in policies.items():
        lines.append(
            f"| `{name}` | {value['true_positive']} | {value['true_negative']} | "
            f"{len(value['false_merges'])} | {len(value['misses'])} |"
        )
    lines += ["", "## Failure cases", ""]
    for name, value in policies.items():
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append("- False merges: " + (", ".join(value["false_merges"]) or "none in this cohort"))
        lines.append("- Misses: " + (", ".join(value["misses"]) or "none in this cohort"))
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Chromaprint is useful as a recording-candidate filter, not as rendition proof.",
        "A five-second tail trim can score `1.0` because the matcher normalizes by the",
        "shorter fingerprint. Mutual coverage helps but overlaps low-bitrate transcodes.",
        "",
        "The post-hoc composite was chosen after seeing this packet. Its perfect row is",
        "training fit, not validation and not a shipping threshold. Controlled codec outputs",
        "do not represent every encoder, damaged file,",
        "or independently sourced copy, and subtle mastering can overlap codec artifacts.",
        "False negatives should remain separate files; false merges violate the promise.",
        "Exact native-rate, native-channel decoded PCM equality is the only tested grouping",
        "rule that extends beyond identical container bytes without a learned threshold.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--fpcalc", type=Path, default=DEFAULT_FPCALC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.fpcalc.is_file():
        raise FileNotFoundError(args.fpcalc)
    args.output.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    records: list[dict[str, object]] = []
    for spec in REAL_PAIRS:
        left_database_path, left_path = track_path(connection, spec.left_track_id)
        right_database_path, right_path = track_path(connection, spec.right_track_id)
        try:
            evidence = compare_pair(
                args.fpcalc,
                left_path,
                right_path,
                f"track-{spec.left_track_id}",
                f"track-{spec.right_track_id}",
                args.output,
                include_pcm=spec.expectation in {"same_rendition", "ambiguous"},
            )
        except subprocess.CalledProcessError as error:
            evidence = {
                "decoder_error": str(error),
                "decoder_return_code": error.returncode,
                "policy_scored": False,
            }
        records.append(
            {
                "name": spec.name,
                "source": "real_library",
                "expectation": spec.expectation,
                "subtype": spec.subtype,
                "rationale": spec.rationale,
                "left": {"track_id": spec.left_track_id, "database_path": left_database_path},
                "right": {"track_id": spec.right_track_id, "database_path": right_database_path},
                "evidence": evidence,
            }
        )
        atomic_json(args.output / "partial-results.json", records)

    derived_dir = args.output / "derived-audio"
    for source in CONTROLLED_SOURCES:
        source_database_path, source_path = track_path(connection, source.track_id)
        for transform in TRANSFORMS:
            derived_path = derived_dir / f"track-{source.track_id}-{transform.name}.{transform.extension}"
            run_ffmpeg_atomic(source_path, derived_path, transform.ffmpeg_args)
            evidence = compare_pair(
                args.fpcalc,
                source_path,
                derived_path,
                f"track-{source.track_id}",
                f"track-{source.track_id}-{transform.name}",
                args.output,
                include_pcm=True,
            )
            records.append(
                {
                    "name": f"controlled_{source.track_id}_{transform.name}",
                    "source": "controlled_transform",
                    "expectation": transform.expectation,
                    "subtype": transform.subtype,
                    "rationale": source.description,
                    "left": {"track_id": source.track_id, "database_path": source_database_path},
                    "right": {"derived_path": str(derived_path)},
                    "evidence": evidence,
                }
            )
            atomic_json(args.output / "partial-results.json", records)

    result = {
        "experiment_version": EXPERIMENT_VERSION,
        "database": str(args.db),
        "database_sha256": sha256_file(args.db),
        "fpcalc": identity_v1.fpcalc_version(args.fpcalc),
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, text=True
        ).stdout.splitlines()[0],
        "records": records,
        "policies": policy_table(records),
        "policy_specs": {
            "byte_identical": "whole-file bytes have identical SHA-256",
            "decoded_pcm_identical": "sample rate and channel layout match and full decoded pcm_f64le streams have identical SHA-256",
            "chromaprint_0_8": "aligned Chromaprint score >= 0.8",
            "chromaprint_0_8_mutual_0_99": "score >= 0.8 and matching frames / longer fingerprint >= 0.99",
            "post_hoc_composite_not_validated": "score >= 0.989, decoded duration delta <= 3 s, aligned mono-8-kHz waveform |r| >= 0.994, and absolute median one-second-block gain delta <= 0.5 dB",
        },
        "decision": {
            "chromaprint_role": "candidate generation only",
            "shipping_threshold": None,
            "default_on_ambiguous_evidence": "keep files as distinct renditions",
            "next_gate": "natural same-master cross-codec labels plus subtle real remaster/edit negatives and independent implementation parity",
        },
    }
    atomic_json(args.output / "results.json", result)
    write_summary(args.output / "summary.md", result)
    (args.output / "partial-results.json").unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "policies": result["policies"]}, indent=2))


if __name__ == "__main__":
    main()
