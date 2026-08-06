#!/usr/bin/env python3
"""Generate pinned PyTorch references for the V2 Android audio parity gate.

The expected vectors come from the production desktop CLaMP3 pipeline in
FP32, without a duration cap. The manifest also binds the source bytes, the
PyTorch checkpoints, the exported Android models, and every sample/window/
segment count that the Android test must reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Required by deterministic CUDA GEMM before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoModel, Wav2Vec2FeatureExtractor


REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_ROOT = REPO_ROOT / "desktop-indexer"
sys.path.insert(0, str(DESKTOP_ROOT / "src"))

from poweramp_indexer.embeddings_clamp3 import (  # noqa: E402
    AUDIO_HIDDEN_SIZE,
    CLAMP3_WEIGHTS_FILENAME,
    MAX_AUDIO_LENGTH,
    MERT_SR,
    WINDOW_SAMPLES,
    CLaMP3AudioEncoder,
    CLaMP3EmbeddingGenerator,
)


MERT_REPOSITORY = "m-a-p/MERT-v1-95M"
MERT_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MERT_ARTIFACT = "pytorch_model.bin"
CLAMP3_REPOSITORY = "sander-wood/clamp3"
CLAMP3_REVISION = "355625cc1c6f73726bbcd0eb9276ac7152d56426"

EMBEDDING_DIM = 768
MINIMUM_TAIL_SAMPLES = MERT_SR
FULL_TRACK_MAX_SECONDS = 2**31 - 1
POLICY_ID = (
    "desktop-clamp3-pytorch-audio-parity-v1:fp32-full-track:"
    "torchaudio-hann-width6-rolloff0.99:pcm24k-whole-track-zmuv:"
    "5s-window:1s-tail-zero-pad:zero-bookends:segment128-final-overlap:"
    "frame-weighted-average:l2"
)
ANDROID_PREPROCESSING_SPEC_ID = (
    "mert-clamp3-audio-v3:torchaudio-hann-v1-width6-rolloff0.99-f32-target-length:"
    "pcm24k-whole-span-zmuv:5s-window:1s-tail-zero-pad:"
    "zero-bookends:segment128-final-overlap:frame-weighted-average:l2"
)
ANDROID_BACKEND_POLICY_ID = (
    "litert-2.1.1-compiled-model-v1:"
    "mert-gpu-fp32-strict:clamp3-audio-gpu-fp32-strict:no-backend-fallback"
)

DEFAULT_FIXTURE_DIR = (
    DESKTOP_ROOT / "audit_raw_data" / "v2-discovery" / "audio-device-parity"
)
DEFAULT_MODELS_DIR = DESKTOP_ROOT / "models"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, encoded)


def torchaudio_target_length(source_samples: int, source_rate: int) -> int:
    """Match TorchAudio 2.10's ceil(float32(new * length / old)) rule."""
    if source_samples < 0 or source_rate <= 0:
        raise ValueError("invalid source sample plan")
    if source_rate == MERT_SR:
        return source_samples
    divisor = math.gcd(source_rate, MERT_SR)
    up = MERT_SR // divisor
    down = source_rate // divisor
    scalar = np.float32((up * source_samples) / down)
    return math.ceil(float(scalar))


def planned_work(canonical_samples: int) -> tuple[int, int, int, bool]:
    if canonical_samples < 0:
        raise ValueError("canonical_samples must be non-negative")
    full_windows, tail_samples = divmod(canonical_samples, WINDOW_SAMPLES)
    tail_included = tail_samples >= MINIMUM_TAIL_SAMPLES
    windows = full_windows + int(tail_included)
    segments = math.ceil((windows + 2) / MAX_AUDIO_LENGTH) if windows else 0
    return windows, segments, tail_samples, tail_included


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="PyTorch reference backend; all paths remain FP32",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional source basename to generate; repeatable",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_determinism() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def load_generator(
    device: torch.device,
    batch_size: int,
) -> tuple[CLaMP3EmbeddingGenerator, Path, Path, str | None]:
    mert_artifact = Path(
        hf_hub_download(
            MERT_REPOSITORY,
            MERT_ARTIFACT,
            revision=MERT_REVISION,
        )
    )
    clamp3_artifact = Path(
        hf_hub_download(
            CLAMP3_REPOSITORY,
            CLAMP3_WEIGHTS_FILENAME,
            revision=CLAMP3_REVISION,
        )
    )

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=MERT_SR,
        padding_value=0.0,
        return_attention_mask=True,
        do_normalize=True,
    )
    mert = AutoModel.from_pretrained(
        MERT_REPOSITORY,
        revision=MERT_REVISION,
        trust_remote_code=True,
    )
    mert.to(device).float().eval()
    for parameter in mert.parameters():
        parameter.requires_grad = False

    clamp3 = CLaMP3AudioEncoder.from_clamp3_checkpoint(
        clamp3_artifact,
        device=str(device),
    ).float().eval()

    generator = CLaMP3EmbeddingGenerator(
        max_duration=FULL_TRACK_MAX_SECONDS,
        batch_size=batch_size,
        fp16=False,
    )
    generator.device = str(device)
    # Inject the pinned production components so lazy loading cannot follow a moved repo ref.
    generator._mert_processor = processor
    generator._mert_model = mert
    generator._clamp3_encoder = clamp3
    return generator, mert_artifact, clamp3_artifact, getattr(mert.config, "_commit_hash", None)


def source_plan(path: Path) -> dict[str, Any]:
    waveform, source_rate = torchaudio.load(str(path))
    if waveform.ndim != 2 or waveform.shape[0] <= 0 or waveform.shape[1] <= 0:
        raise ValueError(f"invalid decoded waveform for {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite decoded waveform for {path}")
    source_samples = int(waveform.shape[1])
    canonical_samples = torchaudio_target_length(source_samples, int(source_rate))
    windows, segments, tail_samples, tail_included = planned_work(canonical_samples)
    return {
        "source_sample_rate_hz": int(source_rate),
        "source_channel_count": int(waveform.shape[0]),
        "source_sample_count": source_samples,
        "provider_duration_ms": source_samples * 1000 // int(source_rate),
        "canonical_sample_count_24k": canonical_samples,
        "mert_windows": windows,
        "clamp_segments": segments,
        "tail_samples_24k": tail_samples,
        "tail_included": tail_included,
    }


def generate_track(
    generator: CLaMP3EmbeddingGenerator,
    source: Path,
    fixture_dir: Path,
) -> dict[str, Any]:
    name = source.stem
    plan = source_plan(source)

    started = time.perf_counter()
    features = generator.extract_mert_features(source)
    mert_ms = round((time.perf_counter() - started) * 1000)
    if features is None or features.shape != (1, plan["mert_windows"], AUDIO_HIDDEN_SIZE):
        shape = None if features is None else tuple(features.shape)
        raise RuntimeError(
            f"{source.name}: production MERT returned {shape}, "
            f"expected (1, {plan['mert_windows']}, {AUDIO_HIDDEN_SIZE})"
        )
    if not np.isfinite(features).all():
        raise RuntimeError(f"{source.name}: production MERT returned non-finite features")

    started = time.perf_counter()
    embedding_list = generator.encode_mert_features(features)
    clamp_ms = round((time.perf_counter() - started) * 1000)
    if embedding_list is None:
        raise RuntimeError(f"{source.name}: production CLaMP3 returned no embedding")
    embedding = np.asarray(embedding_list, dtype="<f4")
    if embedding.shape != (EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise RuntimeError(f"{source.name}: invalid embedding {embedding.shape}")
    norm = float(np.linalg.norm(embedding.astype(np.float64)))
    if abs(norm - 1.0) > 1e-5:
        raise RuntimeError(f"{source.name}: embedding norm is {norm}")

    relative_expected = Path("expected") / f"{name}.f32le"
    expected_path = fixture_dir / relative_expected
    atomic_write_bytes(expected_path, embedding.tobytes(order="C"))
    row = {
        "name": name,
        "source_file": str(Path("source") / source.name),
        "source_sha256": sha256(source),
        "source_size_bytes": source.stat().st_size,
        **plan,
        "embedding_file": str(relative_expected),
        "embedding_sha256": sha256(expected_path),
        "embedding_dimension": EMBEDDING_DIM,
        "embedding_l2_norm": norm,
    }
    print(
        f"{name}: samples24k={plan['canonical_sample_count_24k']} "
        f"windows={plan['mert_windows']} segments={plan['clamp_segments']} "
        f"mert={mert_ms}ms clamp={clamp_ms}ms sha256={row['embedding_sha256']}",
        flush=True,
    )
    return row


def validate_fixture_manifest(manifest: dict[str, Any], fixture_dir: Path) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("policy_id") != POLICY_ID:
        raise ValueError("unexpected audio parity manifest schema/policy")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("audio parity manifest has no tracks")
    names: set[str] = set()
    for track in tracks:
        name = track.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("audio parity track names must be unique and non-empty")
        names.add(name)
        source = fixture_dir / track["source_file"]
        expected = fixture_dir / track["embedding_file"]
        if not source.is_file() or sha256(source) != track["source_sha256"]:
            raise ValueError(f"source hash mismatch for {name}")
        if not expected.is_file() or sha256(expected) != track["embedding_sha256"]:
            raise ValueError(f"embedding hash mismatch for {name}")
        if expected.stat().st_size != EMBEDDING_DIM * np.dtype("<f4").itemsize:
            raise ValueError(f"embedding byte length mismatch for {name}")
        expected_work = planned_work(int(track["canonical_sample_count_24k"]))
        actual_work = (
            int(track["mert_windows"]),
            int(track["clamp_segments"]),
            int(track["tail_samples_24k"]),
            bool(track["tail_included"]),
        )
        if actual_work != expected_work:
            raise ValueError(f"planned work mismatch for {name}: {actual_work} != {expected_work}")


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    fixture_dir = args.fixture_dir.resolve()
    source_dir = fixture_dir / "source"
    sources = sorted(source_dir.glob("*.flac"))
    if args.source:
        requested = set(args.source)
        sources = [source for source in sources if source.name in requested]
        missing = requested - {source.name for source in sources}
        if missing:
            raise FileNotFoundError(f"unknown source files: {sorted(missing)}")
    if not sources:
        raise FileNotFoundError(f"no FLAC sources in {source_dir}")

    models_dir = args.models_dir.resolve()
    android_mert = models_dir / "mert.tflite"
    android_clamp3 = models_dir / "clamp3_audio.tflite"
    for model in (android_mert, android_clamp3):
        if not model.is_file():
            raise FileNotFoundError(model)

    configure_determinism()
    device = resolve_device(args.device)
    print(f"Loading pinned production PyTorch pipeline on {device} in FP32", flush=True)
    generator, mert_artifact, clamp3_artifact, loaded_mert_revision = load_generator(
        device,
        args.batch_size,
    )
    if loaded_mert_revision != MERT_REVISION:
        raise RuntimeError(
            f"loaded MERT revision {loaded_mert_revision}, expected {MERT_REVISION}"
        )

    rows = [generate_track(generator, source, fixture_dir) for source in sources]
    manifest = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "android_preprocessing_spec_id": ANDROID_PREPROCESSING_SPEC_ID,
        "android_inference_backend_policy_id": ANDROID_BACKEND_POLICY_ID,
        "full_track": True,
        "host_inference_dtype": "float32",
        "target_sample_rate_hz": MERT_SR,
        "mert_window_samples": WINDOW_SAMPLES,
        "minimum_tail_samples": MINIMUM_TAIL_SAMPLES,
        "clamp_max_frames": MAX_AUDIO_LENGTH,
        "embedding_dimension": EMBEDDING_DIM,
        "host_models": {
            "mert": {
                "repository": MERT_REPOSITORY,
                "revision": MERT_REVISION,
                "artifact": MERT_ARTIFACT,
                "artifact_sha256": sha256(mert_artifact),
            },
            "clamp3_audio": {
                "repository": CLAMP3_REPOSITORY,
                "revision": CLAMP3_REVISION,
                "artifact": CLAMP3_WEIGHTS_FILENAME,
                "artifact_sha256": sha256(clamp3_artifact),
            },
        },
        "device_models": {
            "mert": {
                "file": "mert.tflite",
                "sha256": sha256(android_mert),
            },
            "clamp3_audio": {
                "file": "clamp3_audio.tflite",
                "sha256": sha256(android_clamp3),
            },
        },
        "host_runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torchaudio": torchaudio.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "tracks": rows,
    }
    validate_fixture_manifest(manifest, fixture_dir)
    manifest_path = fixture_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"{manifest_path} sha256={sha256(manifest_path)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
