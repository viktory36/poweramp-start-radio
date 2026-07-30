from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_residual_direction_eval as residual


def test_exact_angle_direction_has_requested_angle_and_great_circle() -> None:
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    target = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    query, available = residual.exact_angle_direction(seed, target, 30.0)
    assert abs(available - 90.0) < 1e-8
    assert abs(residual.exact_seed_angle_degrees(seed, query) - 30.0) < 1e-5
    np.testing.assert_allclose(query, [np.sqrt(3.0) / 2.0, 0.5, 0.0], atol=1e-7)


def test_exact_angle_direction_rejects_past_target_arc() -> None:
    seed = np.asarray([1.0, 0.0], dtype=np.float32)
    target = np.asarray([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="exceeds target arc"):
        residual.exact_angle_direction(seed, target, 91.0)


def test_neutral_residual_is_tangent_and_signed() -> None:
    seed = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    text = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    neutral = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    toward, toward_tangent = residual.neutral_residual_direction(
        seed, text, neutral, 1, 20.0
    )
    away, away_tangent = residual.neutral_residual_direction(
        seed, text, neutral, -1, 20.0
    )
    assert abs(float(np.dot(seed, toward_tangent))) < 1e-7
    np.testing.assert_allclose(away_tangent, -toward_tangent, atol=1e-7)
    assert abs(residual.exact_seed_angle_degrees(seed, toward) - 20.0) < 1e-5
    assert abs(residual.exact_seed_angle_degrees(seed, away) - 20.0) < 1e-5


def test_overlap_reports_membership_and_position_separately() -> None:
    result = residual.overlap([1, 2, 3], [1, 3, 4])
    assert result == {"intersection": 2, "jaccard": 0.5, "same_position": 1}
