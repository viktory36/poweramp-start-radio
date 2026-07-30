#!/usr/bin/env python3
"""Generate debug-only Android parity fixtures from TorchAudio 2.10 defaults."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = (
    REPO_ROOT
    / "android-plugin"
    / "app"
    / "src"
    / "debug"
    / "assets"
    / "resampler_hann_v1"
)
TARGET_RATE = 24_000
FIXTURES = (
    ("general-44100", 44_100, 50_003, 0x44100),
    ("general-48000", 48_000, 52_001, 0x48000),
    ("short-44100", 44_100, 173, 0x44101),
    ("short-48000", 48_000, 31, 0x48001),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def signal(sample_rate: int, sample_count: int, seed: int) -> np.ndarray:
    index = np.arange(sample_count, dtype=np.float64)
    time = index / sample_rate
    chirp_phase = 2.0 * np.pi * (73.0 * time + 0.5 * 8_000.0 * time * time)
    value = (
        0.31 * np.sin(2.0 * np.pi * 997.0 * time + 0.17)
        + 0.19 * np.sin(2.0 * np.pi * 7_913.0 * time + 0.41)
        + 0.11 * np.sin(chirp_phase)
    )
    noise = np.random.default_rng(seed).standard_normal(sample_count) * 0.025
    value += noise
    for position, amplitude in (
        (0, 0.8),
        (1, -0.6),
        (12, 0.5),
        (79, -0.7),
        (80, 0.65),
        (146, -0.55),
        (147, 0.45),
        (sample_count - 2, -0.75),
        (sample_count - 1, 0.9),
    ):
        if 0 <= position < sample_count:
            value[position] += amplitude
    return np.clip(value, -1.0, 1.0).astype("<f4")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    for name, sample_rate, sample_count, seed in FIXTURES:
        source = signal(sample_rate, sample_count, seed)
        source_tensor = torch.from_numpy(source).unsqueeze(0)
        expected = (
            torchaudio.transforms.Resample(sample_rate, TARGET_RATE)(source_tensor)
            .squeeze(0)
            .contiguous()
            .numpy()
            .astype("<f4", copy=False)
        )
        source_bytes = source.tobytes(order="C")
        expected_bytes = expected.tobytes(order="C")
        source_name = f"{name}.input.f32le"
        expected_name = f"{name}.expected.f32le"
        (OUTPUT / source_name).write_bytes(source_bytes)
        (OUTPUT / expected_name).write_bytes(expected_bytes)
        rows.append(
            {
                "name": name,
                "fromRate": sample_rate,
                "toRate": TARGET_RATE,
                "inputFile": source_name,
                "inputSamples": int(source.size),
                "inputSha256": sha256_bytes(source_bytes),
                "expectedFile": expected_name,
                "expectedSamples": int(expected.size),
                "expectedSha256": sha256_bytes(expected_bytes),
                "chunkSchedule": [1, 7, 79, 80, 81, 503, 4_096],
            }
        )

    manifest = {
        "specId": "torchaudio-hann-v1-width6-rolloff0.99-f32-target-length",
        "generator": str(Path(__file__).resolve()),
        "torchVersion": torch.__version__,
        "torchaudioVersion": torchaudio.__version__,
        "resamplingMethod": "sinc_interp_hann",
        "lowpassFilterWidth": 6,
        "rolloff": 0.99,
        "coefficientDtype": "TorchAudio dtype=None: float64 construction then float32 kernel",
        "targetLength": "ceil(float32(reducedTo * inputSamples / reducedFrom))",
        "fixtures": rows,
    }
    manifest_path = OUTPUT / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(manifest_path)
    print(manifest_path)
    print(sha256_bytes(manifest_path.read_bytes()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
