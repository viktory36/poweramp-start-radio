from __future__ import annotations

import numpy as np
import pytest

import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


def make_library(
    embeddings: np.ndarray,
    artists: tuple[str | None, ...] | None = None,
) -> queue_eval.Library:
    values = np.asarray(embeddings, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    count = values.shape[0]
    return queue_eval.Library(
        track_ids=np.arange(1, count + 1, dtype=np.int64),
        embeddings=values,
        artists=artists or tuple(f"artist-{index}" for index in range(count)),
        albums=tuple(None for _ in range(count)),
        titles=tuple(f"track-{index}" for index in range(count)),
        durations_ms=np.full(count, 180_000, dtype=np.int64),
        clusters=np.full(count, -1, dtype=np.int32),
        sources=tuple("test" for _ in range(count)),
        metadata_keys=tuple(f"metadata-{index}" for index in range(count)),
        filename_keys=tuple(f"filename-{index}" for index in range(count)),
        file_paths=tuple(f"/{index}.flac" for index in range(count)),
    )


def phone_config(
    selection_mode: str,
    *,
    drift_enabled: bool = False,
    automatic_dpp: bool = False,
    candidate_pool_size: int = 0,
) -> dict[str, object]:
    return {
        "selectionMode": selection_mode,
        "driftEnabled": drift_enabled,
        "driftMode": "SEED_INTERPOLATION",
        "numTracks": 30,
        "candidatePoolSize": candidate_pool_size,
        "mmrCandidatePoolFraction": 0.125,
        "dppFixedCandidatePoolFraction": 0.4,
        "dppUsesCertifiedFullDomain": automatic_dpp,
        "diversityLambda": 0.7,
        "dppQualityExponent": 2.0,
        "walkRestartAlpha": 0.2,
        "anchorDecay": "EXPONENTIAL",
        "anchorStrength": 0.8,
        "anchorHalfLifeTracks": 5.0,
        "momentumBeta": 0.9,
        "artistLimitsEnabled": True,
        "maxPerArtist": 4,
        "minArtistSpacing": 2,
    }


def test_effective_pool_uses_exact_active_count_and_integer_floor() -> None:
    assert matrix.effective_pool_count(
        80_323, matrix.SelectorConfig(mode="mmr", reach=0.0025)
    ) == 200
    assert matrix.effective_pool_count(
        80_323, matrix.SelectorConfig(mode="mmr", reach=0.05)
    ) == 4_016


def test_phone_config_uses_selector_specific_reach() -> None:
    mmr = matrix.config_from_phone(phone_config("MMR"))
    drift = matrix.config_from_phone(
        phone_config("MMR", drift_enabled=True)
    )
    fixed_dpp = matrix.config_from_phone(phone_config("DPP"))

    assert mmr.reach == 0.125
    assert drift.reach == 0.125
    assert fixed_dpp.reach == 0.4
    assert not fixed_dpp.dpp_uses_certified_full_domain


def test_phone_automatic_dpp_preserves_full_domain_contract() -> None:
    automatic = matrix.config_from_phone(
        phone_config(
            "DPP",
            automatic_dpp=True,
            candidate_pool_size=123,
        )
    )

    assert automatic.reach == 1.0
    assert automatic.candidate_pool_size == 123
    assert automatic.dpp_uses_certified_full_domain
    assert matrix.effective_pool_count(80_323, automatic) == 80_323


def test_phone_config_rejects_legacy_generic_reach() -> None:
    config = phone_config("CLOSEST")
    config["candidatePoolFraction"] = 0.02

    with pytest.raises(ValueError, match="legacy candidatePoolFraction"):
        matrix.config_from_phone(config)


def test_artist_credit_rule_trims_lowercases_and_exempts_blank() -> None:
    assert matrix.normalized_artist("  A Person  ") == "a person"
    assert matrix.normalized_artist("   ") is None
    assert matrix.normalized_artist(None) is None


def test_mmr_lambda_changes_membership_deterministically() -> None:
    library = make_library(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.10, 0.0],
                [0.95, 0.31, 0.0],
                [0.70, 0.00, 0.714],
            ]
        )
    )
    candidates = np.array([1, 2, 3], dtype=np.int64)
    relevance = library.embeddings[candidates] @ library.embeddings[0]
    focused = matrix.select_mmr(
        library,
        candidates,
        relevance,
        matrix.SelectorConfig(mode="mmr", queue_size=2, mmr_lambda=1.0),
    )[0]
    diverse = matrix.select_mmr(
        library,
        candidates,
        relevance,
        matrix.SelectorConfig(mode="mmr", queue_size=2, mmr_lambda=0.0),
    )[0]
    assert focused == [1, 2]
    assert diverse == [1, 3]
    assert matrix.select_mmr(
        library,
        candidates,
        relevance,
        matrix.SelectorConfig(mode="mmr", queue_size=2, mmr_lambda=0.0),
    )[0] == diverse


def test_dpp_quality_exponent_is_a_deterministic_control() -> None:
    library = make_library(
        np.array(
            [
                [1.0, 0.0],
                [0.5, 0.8660254],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        )
    )
    candidates = np.array([0, 1, 2], dtype=np.int64)
    relevance = np.array([0.9, 0.85, 0.2], dtype=np.float32)
    broad = matrix.select_dpp(
        library,
        candidates,
        relevance,
        matrix.SelectorConfig(mode="dpp", queue_size=2, dpp_exponent=0.0),
    )[0]
    focused = matrix.select_dpp(
        library,
        candidates,
        relevance,
        matrix.SelectorConfig(mode="dpp", queue_size=2, dpp_exponent=4.0),
    )[0]
    assert broad == [0, 2]
    assert focused == [0, 1]


def test_absolute_drift_schedule_is_queue_length_independent() -> None:
    config = matrix.SelectorConfig(
        mode="drift",
        drift_mode="SEED_INTERPOLATION",
        anchor_schedule="EXPONENTIAL",
        anchor_strength=0.8,
        anchor_half_life=5.0,
    )
    values = [float(matrix.drift_alpha(config, step)) for step in range(10)]
    assert np.isclose(values[0], 0.8)
    assert np.isclose(values[5], 0.4)
    assert values == [float(matrix.drift_alpha(config, step)) for step in range(10)]


def test_control_churn_distinguishes_order_only_change() -> None:
    unchanged = matrix.queue_change([1, 2, 3], [1, 2, 3])
    reordered = matrix.queue_change([1, 2, 3], [1, 3, 2])
    assert unchanged["exact_order_no_op"]
    assert not reordered["exact_order_no_op"]
    assert reordered["set_jaccard"] == 1.0
    assert reordered["common_prefix_fraction"] < 1.0
