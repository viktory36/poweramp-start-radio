from __future__ import annotations

import copy

import numpy as np
import pytest

import compare_device_feature_acceptance as oracle
import v2_queue_eval as queue_eval
import v2_selection_knob_matrix as matrix


def make_library(embeddings: np.ndarray) -> queue_eval.Library:
    values = np.asarray(embeddings, dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    count = values.shape[0]
    return queue_eval.Library(
        track_ids=np.arange(1, count + 1, dtype=np.int64),
        embeddings=values,
        artists=tuple(f"artist-{index}" for index in range(count)),
        albums=tuple(None for _ in range(count)),
        titles=tuple(f"track-{index}" for index in range(count)),
        durations_ms=np.full(count, 180_000, dtype=np.int64),
        clusters=np.full(count, -1, dtype=np.int32),
        sources=tuple("test" for _ in range(count)),
        metadata_keys=tuple(f"metadata-{index}" for index in range(count)),
        filename_keys=tuple(f"filename-{index}" for index in range(count)),
        file_paths=tuple(f"/{index}.flac" for index in range(count)),
    )


def selector_config(
    selection_mode: str,
    *,
    automatic_dpp: bool = False,
    candidate_pool_size: int = 0,
) -> dict[str, object]:
    return {
        "selectionMode": selection_mode,
        "candidatePoolSize": candidate_pool_size,
        "mmrCandidatePoolFraction": 0.25,
        "dppFixedCandidatePoolFraction": 0.1,
        "dppUsesCertifiedFullDomain": automatic_dpp,
        "numTracks": 2,
        "dppQualityExponent": 1.0,
        "diversityLambda": 0.5,
        "artistLimitsEnabled": False,
        "maxPerArtist": 8,
        "minArtistSpacing": 3,
    }


def test_selector_specific_pool_fields_and_explicit_override() -> None:
    mmr = selector_config("MMR")
    fixed_dpp = selector_config("DPP")
    automatic_dpp = selector_config(
        "DPP",
        automatic_dpp=True,
        candidate_pool_size=17,
    )

    assert oracle.resolved_candidate_pool_size(mmr, "mmr", 30, 1_000) == 250
    assert oracle.resolved_candidate_pool_size(fixed_dpp, "dpp", 30, 1_000) == 100
    assert oracle.resolved_candidate_pool_size(
        {**mmr, "candidatePoolSize": 123}, "mmr", 30, 1_000
    ) == 123
    assert oracle.resolved_candidate_pool_size(
        automatic_dpp, "dpp", 30, 1_000
    ) == 1_000


def test_legacy_generic_pool_fraction_fails_closed() -> None:
    config = selector_config("MMR")
    config["candidatePoolFraction"] = 0.02

    with pytest.raises(ValueError, match="legacy candidatePoolFraction"):
        oracle.validate_clean_selector_config(config)


@pytest.mark.parametrize(
    ("selection_mode", "drift_enabled", "expected"),
    [
        ("CLOSEST", False, "closest"),
        ("MMR", False, "mmr"),
        ("MMR", True, "drift"),
        ("DPP", False, "dpp"),
        ("RANDOM_WALK", False, "graph"),
        ("UNIFORM_SHUFFLE", False, "uniform_shuffle"),
    ],
)
def test_selector_kind_uses_recorded_config_not_case_id(
    selection_mode: str,
    drift_enabled: bool,
    expected: str,
) -> None:
    assert oracle.selector_kind(
        {
            "selectionMode": selection_mode,
            "driftEnabled": drift_enabled,
        }
    ) == expected


def test_selector_kind_rejects_non_mmr_drift() -> None:
    with pytest.raises(ValueError, match="only for MMR"):
        oracle.selector_kind(
            {
                "selectionMode": "DPP",
                "driftEnabled": True,
            }
        )


@pytest.mark.parametrize(
    ("selection_mode", "missing_key"),
    [
        ("MMR", "mmrCandidatePoolFraction"),
        ("DPP", "dppFixedCandidatePoolFraction"),
    ],
)
def test_missing_selector_specific_fraction_fails_closed(
    selection_mode: str,
    missing_key: str,
) -> None:
    config = selector_config(selection_mode)
    del config[missing_key]

    with pytest.raises(ValueError, match=f"missing required {missing_key}"):
        oracle.validate_clean_selector_config(config)


def test_automatic_dpp_replays_beyond_the_fixed_prefix() -> None:
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
    fixed = selector_config("DPP", candidate_pool_size=2)
    automatic = selector_config(
        "DPP",
        automatic_dpp=True,
        candidate_pool_size=2,
    )
    active_mask = np.ones(library.count, dtype=np.bool_)

    fixed_result = oracle.mmr_or_dpp_expectation(
        library, 0, fixed, "dpp", active_mask=active_mask
    )
    automatic_result = oracle.mmr_or_dpp_expectation(
        library, 0, automatic, "dpp", active_mask=active_mask
    )

    assert fixed_result.track_ids == (2, 3)
    assert automatic_result.track_ids == (2, 4)


@pytest.mark.parametrize("quality_exponent", [0.0, 0.5, 2.0])
def test_exact_dpp_replay_matches_canonical_host_selector(
    quality_exponent: float,
) -> None:
    random = np.random.default_rng(20260715)
    embeddings = random.normal(size=(48, 12)).astype(np.float32)
    artists = tuple(f"artist-{index // 3}" for index in range(48))
    library = make_library(embeddings)
    library = queue_eval.Library(
        track_ids=library.track_ids,
        embeddings=library.embeddings,
        artists=artists,
        albums=library.albums,
        titles=library.titles,
        durations_ms=library.durations_ms,
        clusters=library.clusters,
        sources=library.sources,
        metadata_keys=library.metadata_keys,
        filename_keys=library.filename_keys,
        file_paths=library.file_paths,
    )
    relevance = (library.embeddings @ library.embeddings[0]).astype(
        np.float32, copy=False
    )
    positions = np.arange(1, library.count, dtype=np.int64)
    order = np.lexsort((library.track_ids[positions], -relevance[positions]))
    candidates = positions[order]
    candidate_relevance = relevance[candidates]
    config = matrix.SelectorConfig(
        mode="dpp",
        queue_size=12,
        dpp_exponent=quality_exponent,
        artist_limits=True,
        max_per_artist=2,
        artist_spacing=2,
    )

    expected = matrix.select_dpp(
        library,
        candidates,
        candidate_relevance,
        config,
    )[0]
    actual = oracle.select_dpp_exact(
        library,
        candidates,
        candidate_relevance,
        requested=config.queue_size,
        quality_exponent=quality_exponent,
        artist_limits=config.artist_limits,
        max_per_artist=config.max_per_artist,
        min_spacing=config.artist_spacing,
    )

    assert actual == expected


def test_automatic_dpp_evidence_is_checked_against_active_domain() -> None:
    evidence = {
        "completeCandidateDomainCount": 999,
        "initialWorkingCandidateCount": 100,
        "attemptedCandidateCounts": [100, 200],
        "finalWorkingCandidateCount": 200,
        "finalUnseenInitialGainUpperBound": 0.125,
        "usedCompleteCandidateDomain": False,
        "reproducedFullDomainGreedySequence": True,
    }
    run = {
        "config": selector_config("DPP", automatic_dpp=True),
        "dppSelectionEvidence": evidence,
    }

    validated = oracle.validate_dpp_selection_evidence(run, 999)
    assert validated["validated"] is True
    assert validated["attempted_candidate_counts"] == [100, 200]

    wrong_domain = copy.deepcopy(run)
    wrong_domain["dppSelectionEvidence"]["completeCandidateDomainCount"] = 998
    with pytest.raises(ValueError, match="complete domain disagrees"):
        oracle.validate_dpp_selection_evidence(wrong_domain, 999)

    false_claim = copy.deepcopy(run)
    false_claim["dppSelectionEvidence"]["reproducedFullDomainGreedySequence"] = False
    with pytest.raises(ValueError, match="did not claim"):
        oracle.validate_dpp_selection_evidence(false_claim, 999)


def test_fixed_dpp_rejects_automatic_evidence() -> None:
    run = {
        "config": selector_config("DPP", automatic_dpp=False),
        "dppSelectionEvidence": {},
    }
    with pytest.raises(ValueError, match="must not emit"):
        oracle.validate_dpp_selection_evidence(run, 999)


def test_automatic_dpp_requires_evidence() -> None:
    run = {
        "config": selector_config("DPP", automatic_dpp=True),
        "dppSelectionEvidence": None,
    }
    with pytest.raises(ValueError, match="must emit DppSelectionEvidence"):
        oracle.validate_dpp_selection_evidence(run, 999)
