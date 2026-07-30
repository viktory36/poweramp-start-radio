#!/usr/bin/env python3
"""Independently recompute phone acceptance rankings from the frozen embedding DB.

This is not an app unit test. Its inputs are a phone-produced feature report, the exact
active-catalog artifact pulled from that phone, and the immutable 80,421-row database. It
recomputes cosine ranking on the host and reports both the historical all-row result and the
truthful current-library result after applying the phone's exact active-ID domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import v2_queue_eval as queue_eval
import v2_selection_mode_eval as selection_eval


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "2026-07-07T223308+0300_qv7706c3mq"
    / "embeddings.db"
)


@dataclass(frozen=True)
class RankedExpectation:
    track_ids: tuple[int, ...]
    scores: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("active_catalog", type=Path)
    parser.add_argument("--database", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def active_track_ids(path: Path) -> set[int]:
    result: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if fields and fields[0] == "ACTIVE":
            if len(fields) != 4:
                raise ValueError(f"malformed ACTIVE catalog row: {line!r}")
            result.add(int(fields[1]))
    if not result:
        raise ValueError("active catalog contains no ACTIVE rows")
    return result


def best_first(scores: np.ndarray, track_ids: np.ndarray) -> np.ndarray:
    if scores.shape != track_ids.shape:
        raise ValueError("score and track-ID arrays disagree")
    # Exact cosine ties are rare in this corpus. Track ID is a conservative deterministic
    # fallback; the phone's generation-bound tie token is relevant only for bit-equal scores.
    return np.lexsort((track_ids, -scores))


def normalized_artist(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def can_add_artist(
    library: queue_eval.Library,
    selected: Sequence[int],
    candidate: int,
    max_per_artist: int,
    min_spacing: int,
) -> bool:
    artist = normalized_artist(library.artists[candidate])
    if artist is None:
        return True
    if sum(normalized_artist(library.artists[index]) == artist for index in selected) >= max_per_artist:
        return False
    if min_spacing <= 0:
        return True
    return not any(
        normalized_artist(library.artists[index]) == artist
        for index in selected[-min_spacing:]
    )


def reject_legacy_pool_fraction(config: dict[str, object]) -> None:
    if "candidatePoolFraction" in config:
        raise ValueError(
            "legacy candidatePoolFraction is not accepted; use selector-specific reach fields"
        )


def validate_clean_selector_config(config: dict[str, object]) -> None:
    reject_legacy_pool_fraction(config)
    selection_mode = str(config["selectionMode"])
    if selection_mode == "MMR":
        selector_pool_fraction(config, "mmr")
    elif selection_mode == "DPP":
        automatic = certified_full_domain_dpp(config)
        if not automatic:
            selector_pool_fraction(config, "dpp")


def selector_kind(config: dict[str, object]) -> str:
    selection_mode = str(config.get("selectionMode"))
    drift_enabled = config.get("driftEnabled", False)
    if not isinstance(drift_enabled, bool):
        raise ValueError("driftEnabled must be a boolean")
    if drift_enabled and selection_mode != "MMR":
        raise ValueError("drift is defined only for MMR")
    if selection_mode == "CLOSEST":
        return "closest"
    if selection_mode == "MMR":
        return "drift" if drift_enabled else "mmr"
    if selection_mode == "DPP":
        return "dpp"
    if selection_mode == "RANDOM_WALK":
        return "graph"
    if selection_mode == "UNIFORM_SHUFFLE":
        return "uniform_shuffle"
    raise ValueError(f"unknown selection mode {selection_mode!r}")


def required_fraction(config: dict[str, object], key: str) -> float:
    try:
        raw_value = config[key]
    except KeyError as failure:
        raise ValueError(f"phone config is missing required {key}") from failure
    if isinstance(raw_value, bool):
        raise ValueError(f"{key} must be a finite fraction in (0, 1]")
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as failure:
        raise ValueError(f"{key} must be a finite fraction in (0, 1]") from failure
    if not math.isfinite(value) or value <= 0.0 or value > 1.0:
        raise ValueError(f"{key} must be a finite fraction in (0, 1]")
    return value


def certified_full_domain_dpp(config: dict[str, object]) -> bool:
    reject_legacy_pool_fraction(config)
    try:
        value = config["dppUsesCertifiedFullDomain"]
    except KeyError as failure:
        raise ValueError(
            "DPP phone config is missing required dppUsesCertifiedFullDomain"
        ) from failure
    if not isinstance(value, bool):
        raise ValueError("dppUsesCertifiedFullDomain must be a boolean")
    return value


def selector_pool_fraction(config: dict[str, object], mode: str) -> float:
    reject_legacy_pool_fraction(config)
    if mode == "mmr":
        return required_fraction(config, "mmrCandidatePoolFraction")
    if mode == "dpp":
        return required_fraction(config, "dppFixedCandidatePoolFraction")
    raise ValueError(f"unsupported candidate-pool mode {mode}")


def resolved_candidate_pool_size(
    config: dict[str, object],
    mode: str,
    requested: int,
    domain_count: int,
) -> int:
    if domain_count < 0:
        raise ValueError("candidate domain count cannot be negative")
    if mode == "dpp" and certified_full_domain_dpp(config):
        return domain_count

    configured_pool = int(config["candidatePoolSize"])
    if configured_pool < 0:
        raise ValueError("candidatePoolSize cannot be negative")
    if configured_pool > 0:
        return min(configured_pool, domain_count)
    fraction = selector_pool_fraction(config, mode)
    return min(max(100, requested, int(domain_count * fraction)), domain_count)


def select_prefix(
    library: queue_eval.Library,
    ordered_indices: Iterable[int],
    count: int,
    active_mask: np.ndarray | None,
    artist_limits: bool,
    max_per_artist: int,
    min_spacing: int,
) -> list[int]:
    selected: list[int] = []
    for raw_index in ordered_indices:
        index = int(raw_index)
        if active_mask is not None and not bool(active_mask[index]):
            continue
        if artist_limits and not can_add_artist(
            library,
            selected,
            index,
            max_per_artist,
            min_spacing,
        ):
            continue
        selected.append(index)
        if len(selected) == count:
            break
    return selected


def expectation(
    library: queue_eval.Library,
    scores: np.ndarray,
    count: int,
    active_mask: np.ndarray | None,
    exclude_track_id: int | None = None,
    artist_limits: bool = False,
    max_per_artist: int = 8,
    min_spacing: int = 3,
) -> RankedExpectation:
    working = scores.astype(np.float32, copy=True)
    if exclude_track_id is not None:
        positions = np.flatnonzero(library.track_ids == exclude_track_id)
        if positions.size != 1:
            raise ValueError(f"seed track {exclude_track_id} is not unique")
        working[int(positions[0])] = -np.inf
    order = best_first(working, library.track_ids)
    selected = select_prefix(
        library,
        order,
        count,
        active_mask,
        artist_limits,
        max_per_artist,
        min_spacing,
    )
    return RankedExpectation(
        track_ids=tuple(int(library.track_ids[index]) for index in selected),
        scores=tuple(float(working[index]) for index in selected),
    )


def retrieve_pool(
    library: queue_eval.Library,
    seed_position: int,
    pool_size: int,
    active_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    scores = library.embeddings @ library.embeddings[seed_position]
    eligible = np.ones(library.count, dtype=np.bool_)
    eligible[seed_position] = False
    if active_mask is not None:
        eligible &= active_mask
    positions = np.flatnonzero(eligible)
    order = np.lexsort((library.track_ids[positions], -scores[positions]))
    candidates = positions[order[: min(pool_size, positions.size)]]
    return candidates, scores[candidates].astype(np.float32, copy=False)


def selected_expectation(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    selected: Sequence[int],
) -> RankedExpectation:
    relevance_by_position = {
        int(position): float(score)
        for position, score in zip(candidates, relevance)
    }
    return RankedExpectation(
        track_ids=tuple(int(library.track_ids[position]) for position in selected),
        scores=tuple(relevance_by_position[int(position)] for position in selected),
    )


def dpp_artist_codes(
    library: queue_eval.Library,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    codes = np.full(candidates.size, -1, dtype=np.int32)
    mapping: dict[str, int] = {}
    for local, raw_position in enumerate(candidates):
        artist = normalized_artist(library.artists[int(raw_position)])
        if artist is not None:
            codes[local] = mapping.setdefault(artist, len(mapping))
    return codes, len(mapping)


def select_dpp_exact(
    library: queue_eval.Library,
    candidates: np.ndarray,
    relevance: np.ndarray,
    requested: int,
    quality_exponent: float,
    artist_limits: bool,
    max_per_artist: int,
    min_spacing: int,
) -> list[int]:
    """Replay the phone's Float32 greedy DPP over every supplied candidate row."""
    if not math.isfinite(quality_exponent) or quality_exponent < 0.0:
        raise ValueError("dppQualityExponent must be finite and non-negative")
    if candidates.shape != relevance.shape:
        raise ValueError("DPP candidates and relevance scores disagree")
    if candidates.size == 0 or requested <= 0:
        return []

    artist_codes, artist_count = dpp_artist_codes(library, candidates)
    artist_counts = np.zeros(artist_count, dtype=np.int32)
    recent_artists: list[int] = []
    non_negative = np.maximum(relevance, np.float32(0.0))
    if quality_exponent == 1.0:
        quality = non_negative.astype(np.float32, copy=True)
    else:
        quality = np.power(
            non_negative.astype(np.float64), quality_exponent
        ).astype(np.float32)

    limit = min(requested, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    residual = quality * quality
    remaining = np.ones(candidates.size, dtype=np.bool_)
    selected: list[int] = []

    for step in range(limit):
        eligible = remaining.copy()
        if artist_limits:
            known = artist_codes >= 0
            eligible[known] &= artist_counts[artist_codes[known]] < max_per_artist
            if min_spacing > 0 and recent_artists:
                eligible &= ~np.isin(artist_codes, recent_artists)

        gains = np.where(eligible, residual, -np.inf)
        best = int(np.argmax(gains))
        best_gain = float(gains[best])
        if not math.isfinite(best_gain) or best_gain <= 1e-10:
            break

        selected_position = int(candidates[best])
        selected.append(selected_position)
        remaining[best] = False
        if artist_limits:
            artist_code = int(artist_codes[best])
            if artist_code >= 0:
                artist_counts[artist_code] += 1
            if min_spacing > 0:
                recent_artists.append(artist_code)
                if len(recent_artists) > min_spacing:
                    recent_artists.pop(0)

        root = math.sqrt(best_gain)
        similarities = (
            library.embeddings @ library.embeddings[selected_position]
        ).astype(np.float32, copy=False)
        kernel = quality * quality[best] * similarities[candidates]
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / root
        factors[remaining, step] = new_factor[remaining]
        residual[remaining] -= new_factor[remaining] ** 2
        np.maximum(residual, 0.0, out=residual)
        factors[best, step] = root

    return selected


def mmr_or_dpp_expectation(
    library: queue_eval.Library,
    seed_position: int,
    config: dict[str, object],
    mode: str,
    active_mask: np.ndarray | None,
) -> RankedExpectation:
    requested = int(config["numTracks"])
    domain_count = (
        int(active_mask.sum()) if active_mask is not None else library.count
    )
    pool_size = resolved_candidate_pool_size(
        config,
        mode,
        requested,
        domain_count,
    )
    candidates, relevance = retrieve_pool(
        library,
        seed_position,
        pool_size,
        active_mask,
    )
    if mode == "mmr":
        selected, _ = queue_eval.select_mmr(
            library,
            candidates,
            relevance,
            requested,
            lambda_=float(config["diversityLambda"]),
            constraint_aware=bool(config["artistLimitsEnabled"]),
            max_per_artist=int(config["maxPerArtist"]),
            min_spacing=int(config["minArtistSpacing"]),
        )
    elif mode == "dpp":
        selected = select_dpp_exact(
            library,
            candidates,
            relevance,
            requested,
            float(config["dppQualityExponent"]),
            bool(config["artistLimitsEnabled"]),
            int(config["maxPerArtist"]),
            int(config["minArtistSpacing"]),
        )
    else:
        raise ValueError(mode)
    return selected_expectation(library, candidates, relevance, selected)


def normalize_float32(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    norm = np.float32(np.sqrt(np.sum(result * result, dtype=np.float32)))
    if float(norm) < 1e-10:
        return result
    return (result / norm).astype(np.float32, copy=False)


def drift_expectation(
    library: queue_eval.Library,
    seed_position: int,
    config: dict[str, object],
    active_mask: np.ndarray | None,
) -> RankedExpectation:
    requested = int(config["numTracks"])
    domain_count = (
        int(active_mask.sum()) if active_mask is not None else library.count
    )
    pool_size = resolved_candidate_pool_size(
        config,
        "mmr",
        requested,
        domain_count,
    )
    seed = library.embeddings[seed_position]
    query = seed.copy()
    ema_state: np.ndarray | None = None
    selected: list[int] = []
    selected_scores: list[float] = []
    seen = np.zeros(library.count, dtype=np.bool_)
    seen[seed_position] = True

    lambda_ = np.float32(config["diversityLambda"])
    for step in range(requested):
        similarities = library.embeddings @ query
        eligible_pool = ~seen
        if active_mask is not None:
            eligible_pool &= active_mask
        positions = np.flatnonzero(eligible_pool)
        order = np.lexsort((library.track_ids[positions], -similarities[positions]))
        candidates = positions[order[: min(pool_size, positions.size)]]
        if candidates.size == 0:
            break
        relevance = similarities[candidates].astype(np.float32, copy=False)

        eligible = np.fromiter(
            (
                can_add_artist(
                    library,
                    selected,
                    int(candidate),
                    int(config["maxPerArtist"]),
                    int(config["minArtistSpacing"]),
                )
                if bool(config["artistLimitsEnabled"])
                else True
                for candidate in candidates
            ),
            dtype=np.bool_,
            count=candidates.size,
        )
        if not eligible.any():
            break
        if selected:
            pairwise = library.embeddings[candidates] @ library.embeddings[selected].T
            penalty = np.max(pairwise, axis=1).astype(np.float32, copy=False)
            mmr = lambda_ * relevance - (np.float32(1.0) - lambda_) * penalty
            objective = np.where(eligible, mmr, -np.inf)
            chosen_local = int(np.argmax(objective))
        else:
            chosen_local = int(np.flatnonzero(eligible)[0])

        chosen = int(candidates[chosen_local])
        selected.append(chosen)
        selected_scores.append(float(relevance[chosen_local]))
        seen[chosen] = True
        current = library.embeddings[chosen]

        if config["driftMode"] == "SEED_INTERPOLATION":
            schedule = config["anchorDecay"]
            base = np.float32(config["anchorStrength"])
            half_life = np.float32(config["anchorHalfLifeTracks"])
            elapsed = np.float32(step)
            if schedule == "NONE":
                alpha = base
            elif schedule == "LINEAR":
                alpha = base * np.maximum(
                    np.float32(0.0),
                    np.float32(1.0) - elapsed / (np.float32(2.0) * half_life),
                )
            elif schedule == "EXPONENTIAL":
                alpha = base * np.float32(0.5) ** (elapsed / half_life)
            elif schedule == "STEP":
                alpha = base if elapsed < half_life else base * np.float32(0.2)
            else:
                raise ValueError(f"unknown anchor schedule {schedule}")
            query = normalize_float32(
                alpha * seed + (np.float32(1.0) - alpha) * current
            )
            ema_state = current
        elif config["driftMode"] == "MOMENTUM":
            previous = seed if ema_state is None else ema_state
            beta = np.float32(config["momentumBeta"])
            ema_state = normalize_float32(
                beta * previous + (np.float32(1.0) - beta) * current
            )
            query = ema_state
        else:
            raise ValueError(f"unknown drift mode {config['driftMode']}")

    return RankedExpectation(
        track_ids=tuple(int(library.track_ids[index]) for index in selected),
        scores=tuple(selected_scores),
    )


def graph_expectation(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    seed_position: int,
    config: dict[str, object],
    terminals: np.ndarray,
    active_mask: np.ndarray | None,
) -> RankedExpectation:
    order = np.lexsort((graph.track_ids, -terminals))
    selected: list[int] = []
    for raw_position in order:
        position = int(raw_position)
        if position == seed_position or terminals[position] <= 0.0:
            continue
        if active_mask is not None and not bool(active_mask[position]):
            continue
        if bool(config["artistLimitsEnabled"]) and not can_add_artist(
            library,
            selected,
            position,
            int(config["maxPerArtist"]),
            int(config["minArtistSpacing"]),
        ):
            continue
        selected.append(position)
        if len(selected) == int(config["numTracks"]):
            break
    return RankedExpectation(
        track_ids=tuple(int(library.track_ids[index]) for index in selected),
        scores=tuple(float(terminals[index]) for index in selected),
    )


def build_active_graph(
    library: queue_eval.Library,
    graph: selection_eval.Graph,
    active_mask: np.ndarray,
) -> tuple[selection_eval.Graph, np.ndarray, np.ndarray, dict[str, int]]:
    """Mirror ActiveDomainGraphTopologyBuilder over the frozen host artifacts."""
    if not np.array_equal(graph.track_ids, library.track_ids):
        raise ValueError("graph and embedding rows are not identically aligned")
    if active_mask.shape != (library.count,):
        raise ValueError("active graph mask has the wrong shape")

    active_positions = np.flatnonzero(active_mask)
    n, k = graph.neighbors.shape
    if active_positions.size <= k:
        raise ValueError("active graph domain must contain more tracks than its neighbor count")
    old_to_active = np.full(n, -1, dtype=np.int32)
    old_to_active[active_positions] = np.arange(active_positions.size, dtype=np.int32)
    repaired = np.empty((active_positions.size, k), dtype=np.int32)
    affected_rows = 0
    invalidated_slots = 0

    active_track_ids = graph.track_ids[active_positions]
    for active_row, old_row in enumerate(active_positions):
        mapped = old_to_active[graph.neighbors[old_row]]
        invalid = mapped < 0
        if not bool(np.any(invalid)):
            repaired[active_row] = mapped
            continue

        affected_rows += 1
        invalidated_slots += int(np.count_nonzero(invalid))
        similarities = (
            library.embeddings[active_positions] @ library.embeddings[int(old_row)]
        ).astype(np.float32, copy=False)
        similarities[active_row] = -np.inf
        order = np.lexsort((active_track_ids, -similarities))
        repaired[active_row] = order[:k]

    return (
        selection_eval.Graph(
            track_ids=active_track_ids.copy(),
            neighbors=repaired,
            weights=np.ones_like(repaired, dtype=np.float64),
        ),
        active_positions,
        old_to_active,
        {
            "node_count": int(active_positions.size),
            "neighbors_per_node": int(k),
            "affected_row_count": affected_rows,
            "preserved_row_count": int(active_positions.size) - affected_rows,
            "invalidated_slot_count": invalidated_slots,
        },
    )


MASK64 = (1 << 64) - 1
GOLDEN_GAMMA = -7046029254386353131
MIX_MULTIPLIER_1 = -4658895280553007687
MIX_MULTIPLIER_2 = -7723592293110705685


def u64(value: int) -> int:
    return value & MASK64


def signed64(value: int) -> int:
    value &= MASK64
    return value if value < (1 << 63) else value - (1 << 64)


def mix64(value: int) -> int:
    value = u64(value)
    value = u64((value ^ (value >> 30)) * u64(MIX_MULTIPLIER_1))
    value = u64((value ^ (value >> 27)) * u64(MIX_MULTIPLIER_2))
    return signed64(value ^ (value >> 31))


def legacy_shuffle_namespace(activation_binding_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(
        ("legacy-shuffle-fallback-v1\0" + activation_binding_id).encode("utf-8")
    ).digest()
    return (
        int.from_bytes(digest[:8], "big", signed=True),
        int.from_bytes(digest[8:16], "big", signed=True),
    )


def shuffle_expectation(
    library: queue_eval.Library,
    seed_position: int,
    config: dict[str, object],
    activation_binding_id: str,
    active_mask: np.ndarray | None,
) -> RankedExpectation:
    namespace_high, namespace_low = legacy_shuffle_namespace(activation_binding_id)
    shuffle_seed = int(config["shuffleSeed"])
    seed_mix = mix64(shuffle_seed)
    seed_low_mix = mix64(shuffle_seed + GOLDEN_GAMMA)
    ranked: list[tuple[tuple[int, int, int, int, int], int]] = []
    for position, raw_track_id in enumerate(library.track_ids):
        track_id = int(raw_track_id)
        if position == seed_position:
            continue
        if active_mask is not None and not bool(active_mask[position]):
            continue
        identity_high = mix64(namespace_high ^ track_id)
        identity_low = mix64(namespace_low + track_id * GOLDEN_GAMMA)
        priority_high = mix64(identity_high ^ seed_mix)
        priority_low = mix64(identity_low ^ seed_low_mix)
        key = (
            u64(priority_high),
            u64(priority_low),
            u64(identity_high),
            u64(identity_low),
            track_id,
        )
        ranked.append((key, position))
    ranked.sort(key=lambda item: item[0])
    selected = select_prefix(
        library,
        (position for _, position in ranked),
        int(config["numTracks"]),
        active_mask=None,
        artist_limits=bool(config["artistLimitsEnabled"]),
        max_per_artist=int(config["maxPerArtist"]),
        min_spacing=int(config["minArtistSpacing"]),
    )
    similarities = library.embeddings @ library.embeddings[seed_position]
    return RankedExpectation(
        track_ids=tuple(int(library.track_ids[index]) for index in selected),
        scores=tuple(float(similarities[index]) for index in selected),
    )


def candidate_domain_count(
    library: queue_eval.Library,
    seed_position: int,
    active_mask: np.ndarray | None,
) -> int:
    if active_mask is None:
        return library.count - 1
    if active_mask.shape != (library.count,):
        raise ValueError("active candidate mask has the wrong shape")
    if not bool(active_mask[seed_position]):
        raise ValueError("selection seed is outside the active candidate domain")
    return int(active_mask.sum()) - 1


def evidence_int(evidence: dict[str, object], key: str) -> int:
    try:
        value = evidence[key]
    except KeyError as failure:
        raise ValueError(f"DPP selection evidence is missing {key}") from failure
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"DPP selection evidence {key} must be an integer")
    return value


def evidence_bool(evidence: dict[str, object], key: str) -> bool:
    try:
        value = evidence[key]
    except KeyError as failure:
        raise ValueError(f"DPP selection evidence is missing {key}") from failure
    if not isinstance(value, bool):
        raise ValueError(f"DPP selection evidence {key} must be a boolean")
    return value


def validate_dpp_selection_evidence(
    run: dict[str, object],
    expected_complete_candidate_count: int,
) -> dict[str, object]:
    config = run["config"]
    if not isinstance(config, dict):
        raise ValueError("selection run config must be an object")
    automatic = certified_full_domain_dpp(config)
    evidence_value = run.get("dppSelectionEvidence")

    if not automatic:
        if evidence_value is not None:
            raise ValueError("fixed-neighborhood DPP must not emit full-domain evidence")
        return {
            "validated": True,
            "automatic_full_domain": False,
            "evidence_present": False,
        }
    if not isinstance(evidence_value, dict):
        raise ValueError("automatic DPP must emit DppSelectionEvidence")
    evidence = evidence_value

    complete = evidence_int(evidence, "completeCandidateDomainCount")
    initial = evidence_int(evidence, "initialWorkingCandidateCount")
    final = evidence_int(evidence, "finalWorkingCandidateCount")
    attempted_value = evidence.get("attemptedCandidateCounts")
    if not isinstance(attempted_value, list) or not attempted_value:
        raise ValueError("automatic DPP evidence must list attempted candidate counts")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in attempted_value):
        raise ValueError("attempted DPP candidate counts must be integers")
    attempted = [int(value) for value in attempted_value]

    if complete != expected_complete_candidate_count:
        raise ValueError(
            "DPP evidence complete domain disagrees with the host active domain: "
            f"phone={complete}, host={expected_complete_candidate_count}"
        )
    if complete <= 0 or not 1 <= initial <= complete:
        raise ValueError("automatic DPP evidence has an invalid initial domain size")
    if attempted[0] != initial or attempted[-1] != final:
        raise ValueError("automatic DPP evidence endpoints disagree with its attempts")
    if any(left >= right for left, right in zip(attempted, attempted[1:])):
        raise ValueError("automatic DPP attempt sizes must grow strictly")
    if any(value < initial or value > complete for value in attempted):
        raise ValueError("automatic DPP attempt size is outside the complete domain")

    used_complete = evidence_bool(evidence, "usedCompleteCandidateDomain")
    if used_complete != (final == complete):
        raise ValueError("automatic DPP full-domain flag disagrees with its final size")
    if not evidence_bool(evidence, "reproducedFullDomainGreedySequence"):
        raise ValueError("automatic DPP did not claim the full-domain greedy sequence")

    unseen_bound = evidence.get("finalUnseenInitialGainUpperBound")
    if used_complete:
        if unseen_bound is not None:
            raise ValueError("full-domain DPP evidence must not report an unseen bound")
    else:
        if isinstance(unseen_bound, bool) or not isinstance(unseen_bound, (int, float)):
            raise ValueError("certified DPP prefix must report a finite unseen bound")
        if not math.isfinite(float(unseen_bound)) or float(unseen_bound) < 0.0:
            raise ValueError("certified DPP unseen bound must be finite and non-negative")

    return {
        "validated": True,
        "automatic_full_domain": True,
        "evidence_present": True,
        "complete_candidate_domain_count": complete,
        "initial_working_candidate_count": initial,
        "attempted_candidate_counts": attempted,
        "final_working_candidate_count": final,
        "used_complete_candidate_domain": used_complete,
        "final_unseen_initial_gain_upper_bound": unseen_bound,
        "reproduced_full_domain_greedy_sequence": True,
    }


def comparison(
    actual_rows: Sequence[dict[str, object]],
    expected: RankedExpectation,
) -> dict[str, object]:
    actual_ids = tuple(int(row["trackId"]) for row in actual_rows)
    actual_scores = tuple(float(row["score"]) for row in actual_rows)
    first_id_mismatch = next(
        (
            rank + 1
            for rank, (actual, wanted) in enumerate(zip(actual_ids, expected.track_ids))
            if actual != wanted
        ),
        None,
    )
    score_errors = [
        abs(actual - wanted)
        for actual, wanted in zip(actual_scores, expected.scores)
        if math.isfinite(actual) and math.isfinite(wanted)
    ]
    return {
        "actual_track_ids": list(actual_ids),
        "expected_track_ids": list(expected.track_ids),
        "exact_track_id_order": actual_ids == expected.track_ids,
        "first_track_id_mismatch_rank": first_id_mismatch,
        "max_abs_score_error": max(score_errors, default=0.0),
        "mean_abs_score_error": float(np.mean(score_errors)) if score_errors else 0.0,
    }


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    active_ids = active_track_ids(args.active_catalog)
    library, database_sha256 = queue_eval.load_library(args.database)
    if int(report["generation"]["trackCount"]) != library.count:
        raise ValueError("phone report and frozen database track counts disagree")

    id_to_position = {
        int(track_id): index for index, track_id in enumerate(library.track_ids)
    }
    active_mask = np.fromiter(
        (int(track_id) in active_ids for track_id in library.track_ids),
        dtype=np.bool_,
        count=library.count,
    )
    if int(active_mask.sum()) != int(report["activeCatalog"]["activeTrackCount"]):
        raise ValueError("phone report and active-catalog artifact disagree")

    results: list[dict[str, object]] = []
    graph: selection_eval.Graph | None = None
    graph_transition = None
    graph_terminals: dict[tuple[int, float], np.ndarray] = {}
    active_graph: selection_eval.Graph | None = None
    active_graph_positions: np.ndarray | None = None
    old_to_active_graph: np.ndarray | None = None
    active_graph_transition = None
    active_graph_terminals: dict[tuple[int, float], np.ndarray] = {}
    active_graph_repair: dict[str, int] | None = None
    for run in report.get("selectionRuns", []):
        case_id = str(run["caseId"])
        seed_id = int(run["seedTrackId"])
        seed_position = id_to_position[seed_id]
        scores = library.embeddings @ library.embeddings[seed_position]
        config = run["config"]
        if not isinstance(config, dict):
            raise ValueError(f"selection config for {case_id} must be an object")
        validate_clean_selector_config(config)
        kind = selector_kind(config)
        dpp_evidence_validation: dict[str, object] | None = None
        if kind == "dpp":
            dpp_evidence_validation = validate_dpp_selection_evidence(
                run,
                candidate_domain_count(library, seed_position, active_mask),
            )
        elif run.get("dppSelectionEvidence") is not None:
            raise ValueError(f"non-DPP case {case_id} emitted DPP selection evidence")
        if kind == "closest":
            common = dict(
                library=library,
                scores=scores,
                count=int(config["numTracks"]),
                exclude_track_id=seed_id,
                artist_limits=bool(config["artistLimitsEnabled"]),
                max_per_artist=int(config["maxPerArtist"]),
                min_spacing=int(config["minArtistSpacing"]),
            )
            all_rows = expectation(active_mask=None, **common)
            active_rows = expectation(active_mask=active_mask, **common)
        elif kind in {"mmr", "dpp"}:
            all_rows = mmr_or_dpp_expectation(
                library,
                seed_position,
                config,
                kind,
                active_mask=None,
            )
            active_rows = mmr_or_dpp_expectation(
                library,
                seed_position,
                config,
                kind,
                active_mask=active_mask,
            )
        elif kind == "drift":
            all_rows = drift_expectation(
                library,
                seed_position,
                config,
                active_mask=None,
            )
            active_rows = drift_expectation(
                library,
                seed_position,
                config,
                active_mask=active_mask,
            )
        elif kind == "graph":
            if graph is None:
                graph = selection_eval.parse_graph(args.database)
                if not np.array_equal(graph.track_ids, library.track_ids):
                    raise ValueError("graph and embedding rows are not identically aligned")
                graph_transition = selection_eval.build_edge_transition(
                    graph,
                    weighted=False,
                )
                (
                    active_graph,
                    active_graph_positions,
                    old_to_active_graph,
                    active_graph_repair,
                ) = build_active_graph(library, graph, active_mask)
                active_graph_transition = selection_eval.build_edge_transition(
                    active_graph,
                    weighted=False,
                )
            terminal_key = (seed_position, float(config["walkRestartAlpha"]))
            terminals = graph_terminals.get(terminal_key)
            if terminals is None:
                terminals, _ = selection_eval.exact_terminal_distribution(
                    graph,
                    seed_position,
                    terminal_key[1],
                    graph_transition,
                    weighted=False,
                )
                graph_terminals[terminal_key] = terminals
            all_rows = graph_expectation(
                library,
                graph,
                seed_position,
                config,
                terminals,
                active_mask=None,
            )
            assert active_graph is not None
            assert active_graph_positions is not None
            assert old_to_active_graph is not None
            active_seed_position = int(old_to_active_graph[seed_position])
            if active_seed_position < 0:
                raise ValueError(f"graph seed {seed_id} is outside the active library")
            active_terminal_key = (
                active_seed_position,
                float(config["walkRestartAlpha"]),
            )
            compact_terminals = active_graph_terminals.get(active_terminal_key)
            if compact_terminals is None:
                compact_terminals, _ = selection_eval.exact_terminal_distribution(
                    active_graph,
                    active_seed_position,
                    active_terminal_key[1],
                    active_graph_transition,
                    weighted=False,
                )
                active_graph_terminals[active_terminal_key] = compact_terminals
            active_terminals = np.zeros(library.count, dtype=np.float64)
            active_terminals[active_graph_positions] = compact_terminals
            active_rows = graph_expectation(
                library,
                graph,
                seed_position,
                config,
                active_terminals,
                active_mask=None,
            )
        elif kind == "uniform_shuffle":
            activation_binding_id = str(report["generation"]["activationBindingId"])
            all_rows = shuffle_expectation(
                library,
                seed_position,
                config,
                activation_binding_id,
                active_mask=None,
            )
            active_rows = shuffle_expectation(
                library,
                seed_position,
                config,
                activation_binding_id,
                active_mask=active_mask,
            )
        else:
            raise AssertionError(kind)
        results.append(
            {
                "kind": "selection",
                "case_id": case_id,
                "repeat": int(run["repeat"]),
                "seed_track_id": seed_id,
                "all_database_rows": comparison(
                    run["tracks"], all_rows
                ),
                "active_library_only": comparison(
                    run["tracks"], active_rows
                ),
                "dpp_selection_evidence": dpp_evidence_validation,
            }
        )

    for run in report.get("textRuns", []):
        embedding = np.asarray(run["embedding"], dtype=np.float32)
        if embedding.shape != (library.dim,):
            raise ValueError(f"query {run['query']!r} has wrong embedding shape")
        scores = library.embeddings @ embedding
        count = len(run["tracks"])
        results.append(
            {
                "kind": "text",
                "query": run["query"],
                "repeat": int(run["repeat"]),
                "all_database_rows": comparison(
                    run["tracks"],
                    expectation(
                        library,
                        scores,
                        count,
                        active_mask=None,
                    ),
                ),
                "active_library_only": comparison(
                    run["tracks"],
                    expectation(
                        library,
                        scores,
                        count,
                        active_mask=active_mask,
                    ),
                ),
            }
        )

    output = {
        "schema_version": 2,
        "phone_run_id": report["runId"],
        "phone_generation_id": report["generation"]["generationId"],
        "frozen_database_sha256": database_sha256,
        "track_count": library.count,
        "active_track_count": int(active_mask.sum()),
        "active_graph_repair": active_graph_repair,
        "results": results,
        "summary": {
            "evaluated_runs": len(results),
            "exact_all_row_orders": sum(
                bool(item["all_database_rows"]["exact_track_id_order"])
                for item in results
            ),
            "exact_active_orders": sum(
                bool(item["active_library_only"]["exact_track_id_order"])
                for item in results
            ),
            "largest_score_error": max(
                (
                    max(
                        float(item["all_database_rows"]["max_abs_score_error"]),
                        float(item["active_library_only"]["max_abs_score_error"]),
                    )
                    for item in results
                ),
                default=0.0,
            ),
        },
    }
    serialized = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
