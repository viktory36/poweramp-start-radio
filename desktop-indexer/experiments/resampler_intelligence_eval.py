#!/usr/bin/env python3
"""Measure NativeMath Kaiser vs desktop TorchAudio Hann through CLaMP3 retrieval.

The audio decoder and every model stage are shared. Only the 24 kHz resampler
changes. Intermediate MERT features are checkpointed so an interrupted run resumes
without recomputing completed model work. The frozen phone database is opened
read-only and immutable.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as torch_functional
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoModel


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = REPO_ROOT / "desktop-indexer" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import v2_queue_eval as queue_eval
from poweramp_indexer.embeddings_clamp3 import (
    CLAMP3_WEIGHTS_FILENAME,
    CLaMP3EmbeddingGenerator,
)


TARGET_RATE = 24_000
WINDOW_SAMPLES = 120_000
MIN_TAIL_SAMPLES = 24_000
EXPECTED_DIM = 768
DEFAULT_MANIFEST = EXPERIMENT_DIR / "resampler_intelligence_manifest.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "resampler-intelligence"
)
NATIVE_SOURCE = EXPERIMENT_DIR / "native_math_kaiser_reference.c"

# Preregistered decision thresholds. These are deliberately tighter than the
# previously measured phone-vs-desktop median 49/50 neighbor overlap because this
# experiment changes only one deterministic preprocessing stage.
ACCEPTANCE = {
    "embedding_cosine_min": 0.9995,
    "mert_window_cosine_p01_min": 0.9990,
    "top10_overlap_each_min": 8,
    "top50_overlap_each_min": 45,
    "top50_overlap_median_min": 48.0,
    "stored_seed_rank_delta_max": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_jsonl_durable(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_jsonl_by_slug(path: Path) -> dict[str, dict[str, object]]:
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
                raise ValueError(f"invalid checkpoint line {line_number}: {error}") from error
            result[str(row["slug"])] = row
    return result


class NativeKaiserResampler:
    def __init__(self, source: Path, build_dir: Path):
        build_dir.mkdir(parents=True, exist_ok=True)
        self.source = source.resolve()
        source_hash = sha256_file(self.source)
        self.library_path = build_dir / f"native_math_kaiser_{source_hash[:16]}.so"
        if not self.library_path.exists():
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-O3",
                    "-std=c11",
                    "-fPIC",
                    "-shared",
                    "-ffp-contract=off",
                    str(self.source),
                    "-lm",
                    "-o",
                    str(self.library_path),
                ],
                check=True,
            )
        self.library = ctypes.CDLL(str(self.library_path))
        self.library.native_math_output_length.argtypes = [
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.library.native_math_output_length.restype = ctypes.c_int64
        self.library.native_math_kaiser_resample.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]
        self.library.native_math_kaiser_resample.restype = ctypes.c_int

    def output_length(self, input_length: int, from_rate: int, to_rate: int) -> int:
        result = int(
            self.library.native_math_output_length(input_length, from_rate, to_rate)
        )
        if result < 0:
            raise ValueError("NativeMath output length rejected the input")
        return result

    def resample(self, samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        source = np.ascontiguousarray(samples, dtype=np.float32)
        output_length = self.output_length(source.size, from_rate, to_rate)
        output = np.empty(output_length, dtype=np.float32)
        success = self.library.native_math_kaiser_resample(
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            source.size,
            from_rate,
            to_rate,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output_length,
        )
        if success != 1:
            raise RuntimeError("experiment NativeMath reference rejected resampling")
        return output


@dataclass(frozen=True)
class TrackSpec:
    slug: str
    path: Path
    format: str
    role: str


def load_manifest(path: Path) -> tuple[str, list[TrackSpec]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for raw in payload["tracks"]:
        track = TrackSpec(
            slug=str(raw["slug"]),
            path=Path(raw["path"]),
            format=str(raw["format"]),
            role=str(raw["role"]),
        )
        if not track.path.is_file():
            raise FileNotFoundError(track.path)
        result.append(track)
    if len({track.slug for track in result}) != len(result):
        raise ValueError("manifest slugs must be unique")
    return str(payload["cohort_id"]), result


def decode_mono(path: Path) -> tuple[np.ndarray, int, int]:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError(f"invalid decoded waveform {tuple(waveform.shape)}: {path}")
    channel_count = int(waveform.shape[0])
    mono = waveform.mean(dim=0) if channel_count > 1 else waveform[0]
    result = np.ascontiguousarray(mono.numpy(), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite decoded PCM: {path}")
    return result, int(sample_rate), channel_count


def torchaudio_hann_resample(samples: np.ndarray, from_rate: int) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).unsqueeze(0)
    # These are the defaults used by torchaudio.transforms.Resample in the desktop indexer.
    resampler = torchaudio.transforms.Resample(
        orig_freq=from_rate,
        new_freq=TARGET_RATE,
        resampling_method="sinc_interp_hann",
        lowpass_filter_width=6,
        rolloff=0.99,
        beta=None,
        dtype=None,
    )
    return np.ascontiguousarray(resampler(tensor).squeeze(0).numpy(), dtype=np.float32)


def pcm_metrics(kaiser: np.ndarray, hann: np.ndarray) -> dict[str, object]:
    common_samples = min(kaiser.size, hann.size)
    if common_samples <= 0:
        raise ValueError("resampler produced no common PCM")
    chunk_size = 1_000_000
    dot = 0.0
    left_energy = 0.0
    right_energy = 0.0
    squared_error = 0.0
    absolute_error = 0.0
    maximum_error = 0.0
    for start in range(0, common_samples, chunk_size):
        stop = min(start + chunk_size, common_samples)
        left = kaiser[start:stop].astype(np.float64)
        right = hann[start:stop].astype(np.float64)
        difference = left - right
        dot += float(left @ right)
        left_energy += float(left @ left)
        right_energy += float(right @ right)
        squared_error += float(difference @ difference)
        absolute_error += float(np.abs(difference).sum())
        maximum_error = max(maximum_error, float(np.abs(difference).max(initial=0.0)))
    count = common_samples
    rmse = math.sqrt(squared_error / count)
    reference_rms = math.sqrt(right_energy / count)
    return {
        "kaiser_samples": int(kaiser.size),
        "hann_samples": int(hann.size),
        "length_delta_samples": int(kaiser.size - hann.size),
        "common_samples": int(common_samples),
        "cosine": dot / math.sqrt(left_energy * right_energy),
        "rmse": rmse,
        "relative_rmse": rmse / max(reference_rms, np.finfo(np.float64).tiny),
        "snr_db": 10.0 * math.log10(right_energy / max(squared_error, 1e-300)),
        "mean_absolute_error": absolute_error / count,
        "maximum_absolute_error": maximum_error,
        "kaiser_rms": math.sqrt(left_energy / count),
        "hann_rms": reference_rms,
    }


def normalize_whole_track(samples: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    # Match Wav2Vec2FeatureExtractor's float32 NumPy mean/variance expression.
    mean = samples.mean()
    variance = samples.var()
    normalized = (samples - mean) / np.sqrt(variance + 1e-7)
    return np.ascontiguousarray(normalized, dtype=np.float32), {
        "mean": float(mean),
        "variance": float(variance),
        "standard_deviation_with_epsilon": float(np.sqrt(variance + 1e-7)),
    }


def load_mert(device: torch.device) -> torch.nn.Module:
    model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    if device.type == "cuda":
        model.half()
    return model


def extract_mert_features(
    model: torch.nn.Module,
    samples: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    normalized, normalization = normalize_whole_track(samples)
    full_windows, tail = divmod(normalized.size, WINDOW_SAMPLES)
    window_count = full_windows + (1 if tail >= MIN_TAIL_SAMPLES else 0)
    if window_count == 0:
        raise ValueError("audio is shorter than the one-second MERT minimum")

    features: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch_start in range(0, window_count, batch_size):
        batch_windows = []
        for window_index in range(batch_start, min(batch_start + batch_size, window_count)):
            start = window_index * WINDOW_SAMPLES
            actual = min(WINDOW_SAMPLES, normalized.size - start)
            window = torch.from_numpy(normalized[start : start + actual])
            if actual < WINDOW_SAMPLES:
                window = torch_functional.pad(window, (0, WINDOW_SAMPLES - actual))
            batch_windows.append(window)
        batch = torch.stack(batch_windows).to(device)
        if device.type == "cuda":
            batch = batch.half()
        with torch.inference_mode():
            hidden_states = model(batch, output_hidden_states=True).hidden_states
            # Same reductions as stack(hidden_states).mean(-2).mean(0), without
            # retaining a second full stack of every hidden-state tensor.
            per_layer = torch.stack(
                [hidden_state.mean(dim=-2) for hidden_state in hidden_states], dim=0
            )
            per_window = per_layer.mean(dim=0)
        features.append(per_window.float().cpu())

    merged = torch.cat(features, dim=0).unsqueeze(0).numpy()
    return merged, {
        "window_count": int(window_count),
        "full_windows": int(full_windows),
        "tail_samples": int(tail),
        "tail_included": bool(tail >= MIN_TAIL_SAMPLES),
        "normalization": normalization,
        "inference_seconds": time.perf_counter() - started,
    }


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.ndim != right.ndim or left.shape[-1] != right.shape[-1]:
        raise ValueError(f"incompatible feature shapes: {left.shape} != {right.shape}")
    left_rows = left.reshape(-1, left.shape[-1]).astype(np.float64)
    right_rows = right.reshape(-1, right.shape[-1]).astype(np.float64)
    common_rows = min(left_rows.shape[0], right_rows.shape[0])
    if common_rows <= 0:
        raise ValueError("resampler variants produced no common MERT windows")
    left_rows = left_rows[:common_rows]
    right_rows = right_rows[:common_rows]
    numerator = np.einsum("ij,ij->i", left_rows, right_rows)
    denominator = np.linalg.norm(left_rows, axis=1) * np.linalg.norm(right_rows, axis=1)
    return numerator / denominator


def encode_clamp(generator: CLaMP3EmbeddingGenerator, features: np.ndarray) -> np.ndarray:
    embedding = generator.encode_mert_features(features)
    if embedding is None:
        raise RuntimeError("CLaMP3 returned no embedding")
    result = np.asarray(embedding, dtype=np.float32)
    if result.shape != (EXPECTED_DIM,) or not np.isfinite(result).all():
        raise ValueError(f"invalid CLaMP3 embedding {result.shape}")
    return result


def matching_library_index(library: queue_eval.Library, path: Path) -> int:
    basename = path.name.casefold()
    matches = [
        index
        for index, file_path in enumerate(library.file_paths)
        if file_path.replace("\\", "/").rsplit("/", 1)[-1].casefold() == basename
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen-library basename match for {path}, got {matches}")
    return matches[0]


def stable_top_indices(
    library: queue_eval.Library,
    query: np.ndarray,
    count: int,
    exclude_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = library.embeddings @ query
    if exclude_index is not None:
        similarities[exclude_index] = -np.inf
    count = min(count, library.count - (1 if exclude_index is not None else 0))
    partial = np.argpartition(similarities, -count)[-count:]
    order = np.lexsort((library.track_ids[partial], -similarities[partial]))
    return partial[order].astype(np.int64, copy=False), similarities


def stable_rank(library: queue_eval.Library, similarities: np.ndarray, index: int) -> int:
    score = similarities[index]
    better = int(np.count_nonzero(similarities > score))
    tied_before = int(
        np.count_nonzero(
            (similarities == score) & (library.track_ids < library.track_ids[index])
        )
    )
    return better + tied_before + 1


def overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return len(set(int(value) for value in left) & set(int(value) for value in right))


def rank_displacement(left: Sequence[int], right: Sequence[int]) -> dict[str, float]:
    left_rank = {int(index): rank for rank, index in enumerate(left, start=1)}
    right_rank = {int(index): rank for rank, index in enumerate(right, start=1)}
    missing_rank = max(len(left), len(right)) + 1
    union = sorted(set(left_rank) | set(right_rank))
    displacements = np.asarray(
        [abs(left_rank.get(index, missing_rank) - right_rank.get(index, missing_rank)) for index in union],
        dtype=np.float64,
    )
    return {
        "union_items": int(displacements.size),
        "mean_absolute": float(displacements.mean()),
        "median_absolute": float(np.median(displacements)),
        "maximum_absolute": float(displacements.max(initial=0.0)),
    }


def labels_for_indices(
    library: queue_eval.Library,
    indices: Sequence[int],
    similarities: np.ndarray,
    count: int = 20,
) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "track_id": int(library.track_ids[index]),
            "artist": library.artists[index],
            "title": library.titles[index],
            "score": float(similarities[index]),
        }
        for rank, index in enumerate(indices[:count], start=1)
    ]


def retrieval_metrics(
    library: queue_eval.Library,
    seed_index: int,
    kaiser_embedding: np.ndarray,
    hann_embedding: np.ndarray,
) -> dict[str, object]:
    kaiser_top, kaiser_similarities = stable_top_indices(
        library, kaiser_embedding, 100, exclude_index=seed_index
    )
    hann_top, hann_similarities = stable_top_indices(
        library, hann_embedding, 100, exclude_index=seed_index
    )
    _, kaiser_with_seed = stable_top_indices(library, kaiser_embedding, 1)
    _, hann_with_seed = stable_top_indices(library, hann_embedding, 1)
    stored = library.embeddings[seed_index]
    return {
        "stored_track_id": int(library.track_ids[seed_index]),
        "stored_source": library.sources[seed_index],
        "stored_artist": library.artists[seed_index],
        "stored_title": library.titles[seed_index],
        "kaiser_cosine_to_stored": float(kaiser_embedding @ stored),
        "hann_cosine_to_stored": float(hann_embedding @ stored),
        "kaiser_stored_seed_rank": stable_rank(library, kaiser_with_seed, seed_index),
        "hann_stored_seed_rank": stable_rank(library, hann_with_seed, seed_index),
        "top10_overlap": overlap(kaiser_top[:10], hann_top[:10]),
        "top50_overlap": overlap(kaiser_top[:50], hann_top[:50]),
        "top100_overlap": overlap(kaiser_top, hann_top),
        "top100_rank_displacement": rank_displacement(kaiser_top, hann_top),
        "kaiser_top20": labels_for_indices(library, kaiser_top, kaiser_similarities),
        "hann_top20": labels_for_indices(library, hann_top, hann_similarities),
    }


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    embedding_cosines = [float(row["embedding_cosine"]) for row in rows]
    window_p01 = [float(row["mert"]["window_cosine_p01"]) for row in rows]
    top10 = [int(row["retrieval"]["top10_overlap"]) for row in rows]
    top50 = [int(row["retrieval"]["top50_overlap"]) for row in rows]
    top100 = [int(row["retrieval"]["top100_overlap"]) for row in rows]
    rank_deltas = [
        abs(
            int(row["retrieval"]["kaiser_stored_seed_rank"])
            - int(row["retrieval"]["hann_stored_seed_rank"])
        )
        for row in rows
    ]
    checks = {
        "embedding_cosine_min": min(embedding_cosines) >= ACCEPTANCE["embedding_cosine_min"],
        "mert_window_cosine_p01_min": min(window_p01)
        >= ACCEPTANCE["mert_window_cosine_p01_min"],
        "top10_overlap_each_min": min(top10) >= ACCEPTANCE["top10_overlap_each_min"],
        "top50_overlap_each_min": min(top50) >= ACCEPTANCE["top50_overlap_each_min"],
        "top50_overlap_median_min": float(np.median(top50))
        >= ACCEPTANCE["top50_overlap_median_min"],
        "stored_seed_rank_delta_max": max(rank_deltas)
        <= ACCEPTANCE["stored_seed_rank_delta_max"],
    }
    pass_all = all(checks.values())
    recommendation = "implement_torchaudio_equivalent_hann_on_phone_no_database_reindex"
    return {
        "tracks": len(rows),
        "acceptance_thresholds": ACCEPTANCE,
        "acceptance_checks": checks,
        "passes_all_acceptance_checks": pass_all,
        "current_kaiser_intelligence_status": (
            "accepted_no_material_retrieval_regression"
            if pass_all
            else "rejected_material_retrieval_regression"
        ),
        "recommendation": recommendation,
        "reasoning": (
            "Kaiser passes the preregistered intelligence thresholds, so it is not a demonstrated "
            "recommendation-quality problem. The frozen corpus and desktop indexer already use "
            "TorchAudio Hann, though, and the Hann branch reproduces those stored seed vectors. "
            "V2 should therefore match Hann on phone for cross-indexer determinism. Migrating the "
            "desktop corpus to Kaiser would require a full reindex without demonstrated semantic "
            "benefit."
        ),
        "embedding_cosine": {
            "minimum": min(embedding_cosines),
            "p10": percentile(embedding_cosines, 0.10),
            "median": float(np.median(embedding_cosines)),
            "maximum": max(embedding_cosines),
        },
        "mert_window_cosine_p01": {
            "minimum": min(window_p01),
            "median": float(np.median(window_p01)),
        },
        "top10_overlap": {
            "minimum": min(top10),
            "median": float(np.median(top10)),
            "maximum": max(top10),
        },
        "top50_overlap": {
            "minimum": min(top50),
            "p10": percentile(top50, 0.10),
            "median": float(np.median(top50)),
            "maximum": max(top50),
        },
        "top100_overlap": {
            "minimum": min(top100),
            "median": float(np.median(top100)),
            "maximum": max(top100),
        },
        "stored_seed_rank_delta": {
            "maximum": max(rank_deltas),
            "median": float(np.median(rank_deltas)),
        },
    }


def model_features_for_variant(
    track: TrackSpec,
    variant: str,
    samples: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    cache_dir: Path,
    force: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    feature_path = cache_dir / "mert" / f"{track.slug}.{variant}.npy"
    metadata_path = cache_dir / "mert" / f"{track.slug}.{variant}.json"
    if feature_path.exists() and metadata_path.exists() and not force:
        return np.load(feature_path), json.loads(metadata_path.read_text(encoding="utf-8"))
    features, metadata = extract_mert_features(model, samples, device, batch_size)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = feature_path.with_suffix(".npy.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, features)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(feature_path)
    write_json_atomic(metadata_path, metadata)
    return features, metadata


def evaluate_track(
    track: TrackSpec,
    native: NativeKaiserResampler,
    mert_model: torch.nn.Module,
    clamp_generator: CLaMP3EmbeddingGenerator,
    library: queue_eval.Library,
    device: torch.device,
    batch_size: int,
    cache_dir: Path,
    force: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    source_sha = sha256_file(track.path)
    decoded, source_rate, source_channels = decode_mono(track.path)
    source_samples = int(decoded.size)
    source_duration_seconds = source_samples / source_rate
    if source_rate == TARGET_RATE:
        raise ValueError("cohort must exercise resampling, not a native 24 kHz source")
    print(
        f"  decoded {decoded.size / source_rate:.1f}s @ {source_rate} Hz; resampling",
        flush=True,
    )
    resample_started = time.perf_counter()
    kaiser = native.resample(decoded, source_rate, TARGET_RATE)
    kaiser_seconds = time.perf_counter() - resample_started
    resample_started = time.perf_counter()
    hann = torchaudio_hann_resample(decoded, source_rate)
    hann_seconds = time.perf_counter() - resample_started
    pcm = pcm_metrics(kaiser, hann)
    del decoded

    print(f"  MERT Kaiser ({kaiser.size / TARGET_RATE:.1f}s)", flush=True)
    kaiser_features, kaiser_mert_metadata = model_features_for_variant(
        track, "kaiser", kaiser, mert_model, device, batch_size, cache_dir, force
    )
    del kaiser
    print(f"  MERT Hann ({hann.size / TARGET_RATE:.1f}s)", flush=True)
    hann_features, hann_mert_metadata = model_features_for_variant(
        track, "hann", hann, mert_model, device, batch_size, cache_dir, force
    )
    del hann

    window_cosines = cosine_rows(kaiser_features, hann_features)
    print("  CLaMP and frozen-library retrieval", flush=True)
    kaiser_embedding = encode_clamp(clamp_generator, kaiser_features)
    hann_embedding = encode_clamp(clamp_generator, hann_features)
    seed_index = matching_library_index(library, track.path)
    retrieval = retrieval_metrics(
        library, seed_index, kaiser_embedding, hann_embedding
    )

    embedding_path = cache_dir / "embeddings" / f"{track.slug}.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        embedding_path,
        kaiser=kaiser_embedding,
        hann=hann_embedding,
        mert_window_cosines=window_cosines.astype(np.float32),
    )
    return {
        "slug": track.slug,
        "path": str(track.path),
        "format": track.format,
        "role": track.role,
        "source_sha256": source_sha,
        "source_rate": source_rate,
        "source_channels": source_channels,
        "source_samples": source_samples,
        "source_duration_seconds": source_duration_seconds,
        "pcm": pcm,
        "timing": {
            "kaiser_resample_seconds": kaiser_seconds,
            "hann_resample_seconds": hann_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "mert": {
            "common_windows": int(window_cosines.size),
            "kaiser_windows": int(kaiser_features.shape[1]),
            "hann_windows": int(hann_features.shape[1]),
            "window_cosine_min": float(window_cosines.min()),
            "window_cosine_p01": percentile(window_cosines, 0.01),
            "window_cosine_p10": percentile(window_cosines, 0.10),
            "window_cosine_median": float(np.median(window_cosines)),
            "window_cosine_mean": float(window_cosines.mean()),
            "kaiser": kaiser_mert_metadata,
            "hann": hann_mert_metadata,
        },
        "embedding_cosine": float(kaiser_embedding @ hann_embedding),
        "retrieval": retrieval,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--track", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-db-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    cohort_id, tracks = load_manifest(args.manifest)
    if args.track:
        requested = set(args.track)
        tracks = [track for track in tracks if track.slug in requested]
        missing = requested - {track.slug for track in tracks}
        if missing:
            raise ValueError(f"unknown track slugs: {sorted(missing)}")
    args.output.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output / "cache"
    checkpoint_path = args.output / "track-results.jsonl"
    completed = {} if args.force else load_jsonl_by_slug(checkpoint_path)

    native = NativeKaiserResampler(NATIVE_SOURCE, cache_dir / "native-build")
    library, db_sha = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MERT on {device}", flush=True)
    mert_model = load_mert(device)
    clamp_generator = CLaMP3EmbeddingGenerator(batch_size=args.batch_size, fp16=False)
    clamp_generator.device = str(device)
    clamp_generator._load_clamp3_audio_if_needed()

    for number, track in enumerate(tracks, start=1):
        if track.slug in completed and not args.force:
            print(f"[{number}/{len(tracks)}] {track.slug}: checkpointed", flush=True)
            continue
        print(f"[{number}/{len(tracks)}] {track.slug}", flush=True)
        row = evaluate_track(
            track=track,
            native=native,
            mert_model=mert_model,
            clamp_generator=clamp_generator,
            library=library,
            device=device,
            batch_size=args.batch_size,
            cache_dir=cache_dir,
            force=args.force,
        )
        append_jsonl_durable(checkpoint_path, row)
        completed[track.slug] = row

    selected_rows = [completed[track.slug] for track in tracks if track.slug in completed]
    if len(selected_rows) != len(tracks):
        raise RuntimeError("not every selected track produced a checkpoint")
    summary = summarize(selected_rows)
    mert_artifact = Path(
        hf_hub_download("m-a-p/MERT-v1-95M", "pytorch_model.bin")
    )
    clamp_artifact = Path(
        hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
    )
    android_native_source = (
        REPO_ROOT / "android-plugin" / "app" / "src" / "main" / "cpp" / "math_jni.c"
    )
    environment = {
        "cohort_id": cohort_id,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "native_reference_source_sha256": sha256_file(NATIVE_SOURCE),
        "android_native_math_source_sha256": sha256_file(android_native_source),
        "database": str(args.db.resolve()),
        "database_sha256": db_sha,
        "library_tracks": library.count,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model": "m-a-p/MERT-v1-95M + sander-wood/clamp3",
        "mert_model_revision": getattr(mert_model.config, "_commit_hash", None),
        "mert_model_artifact": str(mert_artifact),
        "mert_model_artifact_sha256": sha256_file(mert_artifact),
        "clamp3_checkpoint": str(clamp_artifact),
        "clamp3_checkpoint_sha256": sha256_file(clamp_artifact),
        "full_track": True,
        "shared_decode": "torchaudio.load then channel mean",
        "kaiser_branch": "scalar extraction of current NativeMath implementation",
        "hann_branch": (
            "torchaudio.transforms.Resample defaults: sinc_interp_hann, width=6, rolloff=.99"
        ),
    }
    write_json_atomic(args.output / "summary.json", summary)
    write_json_atomic(args.output / "environment.json", environment)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
