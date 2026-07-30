from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

import resampler_intelligence_eval as experiment


def test_native_output_length_matches_global_ceil(tmp_path: Path) -> None:
    native = experiment.NativeKaiserResampler(experiment.NATIVE_SOURCE, tmp_path)
    for source_rate in (44_100, 48_000, 96_000):
        for samples in (1, source_rate - 1, source_rate, source_rate + 1, 12_345_678):
            expected = (samples * experiment.TARGET_RATE + source_rate - 1) // source_rate
            assert native.output_length(samples, source_rate, experiment.TARGET_RATE) == expected


def test_native_identity_and_finite_impulse(tmp_path: Path) -> None:
    native = experiment.NativeKaiserResampler(experiment.NATIVE_SOURCE, tmp_path)
    source = np.zeros(44_101, dtype=np.float32)
    source[22_050] = 1.0
    assert np.array_equal(native.resample(source, 44_100, 44_100), source)
    output = native.resample(source, 44_100, experiment.TARGET_RATE)
    assert output.shape == (24_001,)
    assert np.isfinite(output).all()
    assert np.max(np.abs(output)) > 0.1


def test_rank_displacement_uses_missing_rank() -> None:
    metrics = experiment.rank_displacement([1, 2, 3], [1, 3, 4])
    assert metrics == {
        "union_items": 4,
        "mean_absolute": 1.0,
        "median_absolute": 1.0,
        "maximum_absolute": 2.0,
    }


def test_pcm_metrics_identity() -> None:
    samples = np.linspace(-1.0, 1.0, 10_001, dtype=np.float32)
    metrics = experiment.pcm_metrics(samples, samples.copy())
    assert metrics["cosine"] == 1.0
    assert metrics["rmse"] == 0.0
    assert metrics["relative_rmse"] == 0.0
