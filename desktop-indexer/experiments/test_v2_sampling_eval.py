from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import v2_sampling_eval as sampling


def test_uniform_indices_cover_endpoints_and_are_unique() -> None:
    result = sampling.uniform_indices(100, 6)
    assert result.tolist() == [0, 20, 40, 59, 79, 99]
    assert len(set(result.tolist())) == 6


def test_uniform_uses_all_when_budget_exceeds_track() -> None:
    np.testing.assert_array_equal(sampling.uniform_indices(4, 12), np.arange(4))


def test_prefix_policy_is_bounded() -> None:
    np.testing.assert_array_equal(sampling.policy_indices(10, "prefix", 6), np.arange(6))
    np.testing.assert_array_equal(sampling.policy_indices(3, "prefix", 6), np.arange(3))


def test_full_policy_uses_every_window() -> None:
    np.testing.assert_array_equal(sampling.policy_indices(7, "full", None), np.arange(7))


def test_overlap_counts_membership_not_positions() -> None:
    assert sampling.overlap([1, 2, 3], [3, 1, 9]) == 2
