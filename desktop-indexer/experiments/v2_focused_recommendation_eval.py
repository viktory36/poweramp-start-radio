#!/usr/bin/env python3
"""Focused recommendation discovery on one immutable V2 generation.

This evaluator deliberately keeps three product requests separate:

* single seed: compare direct, relevance-preserving diversity, and reciprocal ranks;
* multiple anchors: compare ranked All-of with a varied membership planner; and
* deterministic freshness: inspect later blocks of one full-domain DPP sequence.

Embeddings alone define relevance and membership. Metadata is used only for exact
identity exclusions, explicit artist constraints, display, and post-selection
diagnostics. The script never writes to the input generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sqlite3
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "phone-snapshots"
    / "2026-07-25T1708+0300_qv7706c3mq"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "discovery"
    / "evidence"
    / "focused-recommendation-2026-07-28"
)
EXPECTED_DATABASE_SHA256 = (
    "28fca3bf828efc06bd1d57366f0f7e70ecdf7d72ec0c3d8333d00ec8b06e6eba"
)
EXPECTED_EMBEDDING_SHA256 = (
    "35125e7719a828b09f663ccfcf644e84c68c25c2fa2ca458591e71564f8c89af"
)
EXPECTED_GRAPH_SHA256 = (
    "450fc7def44b470a46de1add5651f4eef4ebb0ea4f8cdbf7685e746060d434aa"
)
EXPECTED_TRACKS = 85_567
EXPECTED_DIM = 768
QUEUE_SIZE = 30
MAX_PER_ARTIST = 8
MIN_ARTIST_SPACING = 3
NEAR_IDENTICAL_COSINE = 0.9999
MIN_DPP_GAIN = 1e-10


@dataclass(frozen=True)
class SeedCase:
    key: str
    track_id: int
    slice_name: str


SEED_CASES: tuple[SeedCase, ...] = (
    SeedCase("radiohead_creep", 239, "mainstream rock"),
    SeedCase("bonobo_cirrus", 5320, "electronic"),
    SeedCase("aphex_rhubarb", 2768, "ambient"),
    SeedCase("kailash_naiharwa", 33821, "non-English"),
    SeedCase("khruangbin_august_10", 16867, "acoustic and global"),
    SeedCase("hallucinogen_demention", 13384, "niche electronic"),
    SeedCase("takeo_moriyama_watarase", 74859, "jazz"),
    SeedCase("brian_regan_phonix", 43529, "sparse spoken neighborhood"),
)


@dataclass(frozen=True)
class MultiAnchorCase:
    key: str
    track_ids: tuple[int, ...]
    text_names: tuple[str, ...] = ()


MULTI_ANCHOR_CASES: tuple[MultiAnchorCase, ...] = (
    MultiAnchorCase("bonobo_and_aphex", (5320, 2768)),
    MultiAnchorCase("nusrat_and_kailash", (209, 33821)),
    MultiAnchorCase("khruangbin_and_nicola_cruz", (16867, 21972)),
    MultiAnchorCase("radiohead_and_tool", (239, 31830)),
    MultiAnchorCase("bonobo_and_ambient_techno", (5320,), ("ambient techno",)),
    MultiAnchorCase(
        "khruangbin_and_moody_guitar",
        (16867,),
        ("moody guitar progressive rock",),
    ),
)

TEXT_QUERIES: tuple[str, ...] = (
    "ambient techno",
    "easy listening 2am",
    "sitar, electronic",
    "slow tamil songs",
    "stand up comedy",
)


@dataclass
class Library:
    track_ids: np.ndarray
    embeddings: np.ndarray
    artists: tuple[str, ...]
    albums: tuple[str, ...]
    titles: tuple[str, ...]
    metadata_keys: tuple[str, ...]
    filename_keys: tuple[str, ...]
    file_paths: tuple[str, ...]
    durations_ms: np.ndarray
    index_by_track_id: dict[int, int]

    @property
    def count(self) -> int:
        return int(self.track_ids.size)


@dataclass(frozen=True)
class Graph:
    neighbors: np.ndarray
    weights: np.ndarray
    indegree: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve()}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    return connection


def load_embedding_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header = handle.read(16)
    if len(header) != 16:
        raise ValueError("embedding header is truncated")
    magic, version, count, dimension = struct.unpack("<4sIII", header)
    if magic != b"PEMB" or version != 1:
        raise ValueError(f"unsupported embedding header: {magic!r}, version {version}")
    if count != EXPECTED_TRACKS or dimension != EXPECTED_DIM:
        raise ValueError(
            f"unexpected embedding shape: {count} x {dimension}, "
            f"expected {EXPECTED_TRACKS} x {EXPECTED_DIM}"
        )
    expected_size = 16 + count * 8 + count * dimension * 4
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"embedding byte length {path.stat().st_size} != expected {expected_size}"
        )
    track_ids = np.memmap(
        path,
        mode="r",
        dtype="<i8",
        offset=16,
        shape=(count,),
    )
    embeddings = np.memmap(
        path,
        mode="r",
        dtype="<f4",
        offset=16 + count * 8,
        shape=(count, dimension),
    )
    return track_ids, embeddings


def load_library(database_path: Path, embedding_path: Path) -> Library:
    track_ids, embeddings = load_embedding_file(embedding_path)
    with read_only_connection(database_path) as connection:
        rows = connection.execute(
            "SELECT id, COALESCE(artist, ''), COALESCE(album, ''), "
            "COALESCE(title, ''), metadata_key, filename_key, file_path, "
            "COALESCE(duration_ms, -1) FROM tracks ORDER BY id"
        )
        ids: list[int] = []
        artists: list[str] = []
        albums: list[str] = []
        titles: list[str] = []
        metadata_keys: list[str] = []
        filename_keys: list[str] = []
        file_paths: list[str] = []
        durations: list[int] = []
        for row in rows:
            ids.append(int(row[0]))
            artists.append(str(row[1]))
            albums.append(str(row[2]))
            titles.append(str(row[3]))
            metadata_keys.append(str(row[4]))
            filename_keys.append(str(row[5]))
            file_paths.append(str(row[6]))
            durations.append(int(row[7]))
    database_ids = np.asarray(ids, dtype=np.int64)
    if not np.array_equal(database_ids, track_ids):
        raise ValueError("database and embedding file do not have the same ordered track IDs")
    if database_ids.size != EXPECTED_TRACKS:
        raise ValueError(f"database has {database_ids.size} tracks, expected {EXPECTED_TRACKS}")
    sampled = np.asarray(embeddings[::997], dtype=np.float32)
    norms = np.linalg.norm(sampled.astype(np.float64), axis=1)
    if not np.isfinite(sampled).all() or not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError("sampled embedding rows are non-finite or non-unit")
    return Library(
        track_ids=track_ids,
        embeddings=embeddings,
        artists=tuple(artists),
        albums=tuple(albums),
        titles=tuple(titles),
        metadata_keys=tuple(metadata_keys),
        filename_keys=tuple(filename_keys),
        file_paths=tuple(file_paths),
        durations_ms=np.asarray(durations, dtype=np.int64),
        index_by_track_id={int(track_id): i for i, track_id in enumerate(track_ids)},
    )


def load_graph(path: Path, library: Library) -> Graph:
    with path.open("rb") as handle:
        header = handle.read(8)
    count, width = struct.unpack("<II", header)
    if count != library.count or width != 5:
        raise ValueError(f"unexpected graph shape: {count} x {width}")
    graph_ids = np.memmap(path, mode="r", dtype="<i8", offset=8, shape=(count,))
    if not np.array_equal(graph_ids, library.track_ids):
        raise ValueError("graph and embedding file do not have the same ordered track IDs")
    entries = np.memmap(
        path,
        mode="r",
        dtype=np.dtype([("neighbor", "<u4"), ("weight", "<f4")]),
        offset=8 + count * 8,
        shape=(count, width),
    )
    neighbors = np.asarray(entries["neighbor"], dtype=np.int64)
    weights = np.asarray(entries["weight"], dtype=np.float32)
    if np.any(neighbors < 0) or np.any(neighbors >= count):
        raise ValueError("graph contains an out-of-range neighbor")
    indegree = np.bincount(neighbors.ravel(), minlength=count).astype(np.int32)
    return Graph(neighbors=neighbors, weights=weights, indegree=indegree)


def stable_order(
    scores: np.ndarray,
    track_ids: np.ndarray,
    domain: np.ndarray | None = None,
) -> np.ndarray:
    candidates = (
        np.arange(scores.size, dtype=np.int64)
        if domain is None
        else np.asarray(domain, dtype=np.int64)
    )
    return candidates[
        np.lexsort((track_ids[candidates], -scores[candidates]))
    ].astype(np.int64, copy=False)


def normalized_artist(value: str) -> str:
    return value.strip().casefold()


def can_add_artist(
    library: Library,
    candidate: int,
    selected: Sequence[int],
    max_per_artist: int = MAX_PER_ARTIST,
    min_spacing: int = MIN_ARTIST_SPACING,
) -> bool:
    artist = normalized_artist(library.artists[candidate])
    if not artist:
        return True
    if sum(normalized_artist(library.artists[index]) == artist for index in selected) >= max_per_artist:
        return False
    return not any(
        normalized_artist(library.artists[index]) == artist
        for index in selected[-min_spacing:]
    )


def identity_exclusions(library: Library, seed_indices: Iterable[int]) -> set[int]:
    result: set[int] = set()
    metadata = {library.metadata_keys[index] for index in seed_indices}
    paths = {library.file_paths[index].casefold() for index in seed_indices}
    for index in range(library.count):
        if (
            library.metadata_keys[index] in metadata
            or library.file_paths[index].casefold() in paths
        ):
            result.add(index)
    return result


def constrained_prefix(
    library: Library,
    ranking: Iterable[int],
    count: int,
    excluded: set[int],
) -> list[int]:
    selected: list[int] = []
    for raw in ranking:
        candidate = int(raw)
        if candidate in excluded or not can_add_artist(library, candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def encode_candidate_artists(
    library: Library,
    candidates: np.ndarray,
) -> tuple[np.ndarray, int]:
    codes = np.full(candidates.size, -1, dtype=np.int32)
    code_by_artist: dict[str, int] = {}
    for local, candidate in enumerate(candidates):
        artist = normalized_artist(library.artists[int(candidate)])
        if not artist:
            continue
        code = code_by_artist.get(artist)
        if code is None:
            code = len(code_by_artist)
            code_by_artist[artist] = code
        codes[local] = code
    return codes, len(code_by_artist)


def artist_eligibility(
    remaining: np.ndarray,
    artist_codes: np.ndarray,
    artist_counts: np.ndarray,
    recent_codes: Sequence[int],
    max_per_artist: int = MAX_PER_ARTIST,
) -> np.ndarray:
    eligible = remaining.copy()
    known = artist_codes >= 0
    eligible[known] &= artist_counts[artist_codes[known]] < max_per_artist
    if recent_codes:
        eligible &= ~np.isin(
            artist_codes,
            np.asarray([code for code in recent_codes if code >= 0], dtype=np.int32),
        )
    return eligible


def select_mmr(
    library: Library,
    relevance: np.ndarray,
    ranking: np.ndarray,
    count: int,
    lambda_: float,
    excluded: set[int],
) -> list[int]:
    candidates = np.asarray(
        [index for index in ranking if int(index) not in excluded],
        dtype=np.int64,
    )
    vectors = np.asarray(library.embeddings[candidates], dtype=np.float32)
    artist_codes, artist_count = encode_candidate_artists(library, candidates)
    artist_counts = np.zeros(artist_count, dtype=np.int32)
    recent_codes: list[int] = []
    remaining = np.ones(candidates.size, dtype=bool)
    maximum_overlap = np.full(candidates.size, -np.inf, dtype=np.float32)
    selected: list[int] = []
    for step in range(min(count, candidates.size)):
        penalty = np.zeros(candidates.size, dtype=np.float32) if step == 0 else maximum_overlap
        objective = np.float32(lambda_) * relevance[candidates] - np.float32(
            1.0 - lambda_
        ) * penalty
        eligible = artist_eligibility(
            remaining,
            artist_codes,
            artist_counts,
            recent_codes,
        )
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, objective, -np.inf)))
        selected.append(int(candidates[best]))
        remaining[best] = False
        artist_code = int(artist_codes[best])
        if artist_code >= 0:
            artist_counts[artist_code] += 1
        recent_codes.append(artist_code)
        if len(recent_codes) > MIN_ARTIST_SPACING:
            recent_codes.pop(0)
        overlap = vectors @ vectors[best]
        maximum_overlap = np.maximum(maximum_overlap, overlap)
    return selected


def dpp_sequence(
    library: Library,
    relevance: np.ndarray,
    ranking: np.ndarray,
    count: int,
    quality_exponent: float,
    excluded: set[int],
    enforce_artist_constraints: bool = True,
) -> tuple[list[int], list[float]]:
    candidates = np.asarray(
        [index for index in ranking if int(index) not in excluded],
        dtype=np.int64,
    )
    vectors = np.asarray(library.embeddings[candidates], dtype=np.float32)
    artist_codes, artist_count = encode_candidate_artists(library, candidates)
    artist_counts = np.zeros(artist_count, dtype=np.int32)
    recent_codes: list[int] = []
    quality = np.maximum(relevance[candidates], 0.0) ** np.float32(quality_exponent)
    limit = min(count, candidates.size)
    factors = np.zeros((candidates.size, limit), dtype=np.float32)
    diagonal = quality * quality
    remaining = np.ones(candidates.size, dtype=bool)
    selected: list[int] = []
    gains: list[float] = []
    for step in range(limit):
        eligible = (
            artist_eligibility(
                remaining,
                artist_codes,
                artist_counts,
                recent_codes,
            )
            if enforce_artist_constraints
            else remaining.copy()
        )
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, diagonal, -np.inf)))
        gain = float(diagonal[best])
        if not math.isfinite(gain) or gain <= MIN_DPP_GAIN:
            break
        selected.append(int(candidates[best]))
        gains.append(gain)
        remaining[best] = False
        artist_code = int(artist_codes[best])
        if enforce_artist_constraints:
            if artist_code >= 0:
                artist_counts[artist_code] += 1
            recent_codes.append(artist_code)
            if len(recent_codes) > MIN_ARTIST_SPACING:
                recent_codes.pop(0)
        sqrt_gain = math.sqrt(gain)
        kernel = quality * quality[best] * (vectors @ vectors[best])
        if step:
            kernel -= factors[:, :step] @ factors[best, :step]
        new_factor = kernel / np.float32(sqrt_gain)
        factors[remaining, step] = new_factor[remaining]
        diagonal[remaining] -= new_factor[remaining] ** 2
        np.maximum(diagonal, 0.0, out=diagonal)
        factors[best, step] = np.float32(sqrt_gain)
    return selected, gains


def reciprocal_reverse_ranks(
    library: Library,
    seed_index: int,
    candidates: np.ndarray,
    domain: np.ndarray | None = None,
) -> np.ndarray:
    """Return exact 1-based rank of the seed from each candidate.

    Float32 scores greater than the candidate/seed score are counted. Equal scores
    use ascending track ID, matching the stable complete-domain order.
    """
    domain_indices = (
        np.arange(library.count, dtype=np.int64)
        if domain is None
        else np.asarray(domain, dtype=np.int64)
    )
    domain_vectors = np.asarray(library.embeddings[domain_indices], dtype=np.float32)
    seed_id = int(library.track_ids[seed_index])
    result = np.empty(candidates.size, dtype=np.int64)
    batch_size = 32
    for start in range(0, candidates.size, batch_size):
        end = min(start + batch_size, candidates.size)
        batch = candidates[start:end]
        similarities = np.asarray(
            library.embeddings[batch], dtype=np.float32
        ) @ domain_vectors.T
        seed_scores = similarities[:, np.flatnonzero(domain_indices == seed_index)[0]]
        greater = np.count_nonzero(similarities > seed_scores[:, None], axis=1)
        tied_lower_id = np.count_nonzero(
            (similarities == seed_scores[:, None])
            & (library.track_ids[domain_indices][None, :] < seed_id),
            axis=1,
        )
        result[start:end] = greater + tied_lower_id + 1
    return result


def mutual_rankings(
    library: Library,
    seed_index: int,
    relevance_order: np.ndarray,
    excluded: set[int],
    count: int,
) -> tuple[dict[str, list[int]], dict[str, object]]:
    """Build exact certified mutual rankings from an expanding forward prefix."""
    usable = np.asarray(
        [index for index in relevance_order if int(index) not in excluded],
        dtype=np.int64,
    )
    prefix = min(128, usable.size)
    timings: list[dict[str, object]] = []
    while True:
        candidates = usable[:prefix]
        started = time.perf_counter()
        reverse = reciprocal_reverse_ranks(library, seed_index, candidates)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        forward = np.arange(1, prefix + 1, dtype=np.int64)
        formulas = {
            "mutual_minimax": np.lexsort(
                (
                    library.track_ids[candidates],
                    forward + reverse,
                    np.maximum(forward, reverse),
                )
            ),
            "mutual_product": np.lexsort(
                (
                    library.track_ids[candidates],
                    np.maximum(forward, reverse),
                    forward * reverse,
                )
            ),
            "mutual_rrf": np.lexsort(
                (
                    library.track_ids[candidates],
                    -(
                        1.0 / (60.0 + forward.astype(np.float64))
                        + 1.0 / (60.0 + reverse.astype(np.float64))
                    ),
                )
            ),
        }
        selected = {
            name: constrained_prefix(
                library,
                candidates[local_order],
                count,
                excluded=set(),
            )
            for name, local_order in formulas.items()
        }
        # For minimax and product, an unseen row has forward rank > prefix.
        # If every accepted kth key is strictly below that bound, the prefix is
        # a complete-domain certificate. RRF is retained as a bounded challenger.
        minimax_positions = [
            int(np.flatnonzero(candidates == item)[0])
            for item in selected["mutual_minimax"]
        ]
        product_positions = [
            int(np.flatnonzero(candidates == item)[0])
            for item in selected["mutual_product"]
        ]
        minimax_bound = max(
            (
                max(int(forward[position]), int(reverse[position]))
                for position in minimax_positions
            ),
            default=sys.maxsize,
        )
        product_bound = max(
            (
                int(forward[position]) * int(reverse[position])
                for position in product_positions
            ),
            default=sys.maxsize,
        )
        minimax_certified = len(minimax_positions) == count and minimax_bound < prefix + 1
        product_certified = (
            len(product_positions) == count
            and product_bound < (prefix + 1) * 1
        )
        timings.append(
            {
                "prefix": prefix,
                "reverseRankMs": elapsed_ms,
                "minimaxCertified": minimax_certified,
                "productCertified": product_certified,
            }
        )
        if minimax_certified and product_certified:
            return selected, {
                "prefix": prefix,
                "timings": timings,
                "forwardRanks": forward.tolist(),
                "reverseRanks": reverse.tolist(),
                "candidateTrackIds": [
                    int(library.track_ids[index]) for index in candidates
                ],
                "rrfDomain": (
                    f"top {prefix} direct matches; RRF is not complete-domain certified"
                ),
            }
        if prefix == usable.size:
            return selected, {
                "prefix": prefix,
                "timings": timings,
                "forwardRanks": forward.tolist(),
                "reverseRanks": reverse.tolist(),
                "candidateTrackIds": [
                    int(library.track_ids[index]) for index in candidates
                ],
                "rrfDomain": "complete domain",
            }
        prefix = min(usable.size, prefix * 2)


def tie_aware_upper_percentile(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    result = np.empty(scores.size, dtype=np.float32)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        # Fraction at or below this tied score. The maximum therefore equals 1.
        result[order[start:end]] = np.float32(end / scores.size)
        start = end
    return result


def all_of_objective(
    library: Library,
    anchors: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    percentiles: list[np.ndarray] = []
    for anchor in anchors:
        raw = np.asarray(library.embeddings @ anchor, dtype=np.float32)
        percentiles.append(tie_aware_upper_percentile(raw))
    matrix = np.stack(percentiles)
    objective = np.exp(
        np.mean(np.log(np.maximum(matrix, np.float32(1.0 / library.count))), axis=0)
    ).astype(np.float32)
    return objective, matrix


def rank_positions(order: np.ndarray) -> np.ndarray:
    result = np.empty(order.size, dtype=np.int64)
    result[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return result


def track_label(library: Library, index: int) -> dict[str, object]:
    return {
        "trackId": int(library.track_ids[index]),
        "artist": library.artists[index],
        "title": library.titles[index],
        "album": library.albums[index],
        "durationMs": int(library.durations_ms[index]),
    }


def queue_metrics(
    library: Library,
    graph: Graph,
    selected: Sequence[int],
    relevance: np.ndarray,
    relevance_positions: np.ndarray,
    anchor_percentiles: np.ndarray | None = None,
) -> dict[str, object]:
    indices = np.asarray(selected, dtype=np.int64)
    if indices.size == 0:
        return {"returned": 0}
    vectors = np.asarray(library.embeddings[indices], dtype=np.float32)
    pairwise = vectors @ vectors.T
    upper = pairwise[np.triu_indices(indices.size, 1)]
    adjacent = np.sum(vectors[:-1] * vectors[1:], axis=1)
    metadata_excess = len(selected) - len(
        {library.metadata_keys[index] for index in selected}
    )
    artist_title_excess = len(selected) - len(
        {
            (
                normalized_artist(library.artists[index]),
                library.titles[index].strip().casefold(),
            )
            for index in selected
        }
    )
    near_identical_excess = 0
    for position in range(indices.size):
        if position and float(np.max(pairwise[position, :position])) >= NEAR_IDENTICAL_COSINE:
            near_identical_excess += 1
    top500 = stable_order(relevance, library.track_ids)[: min(500, library.count)]
    coverage = np.max(
        np.asarray(library.embeddings[top500], dtype=np.float32) @ vectors.T,
        axis=1,
    )
    result: dict[str, object] = {
        "returned": int(indices.size),
        "meanRelevance": float(np.mean(relevance[indices])),
        "p05Relevance": float(np.quantile(relevance[indices], 0.05)),
        "minimumRelevance": float(np.min(relevance[indices])),
        "medianGlobalRank": float(np.median(relevance_positions[indices])),
        "maximumGlobalRank": int(np.max(relevance_positions[indices])),
        "meanPairwiseCosine": float(np.mean(upper)) if upper.size else 1.0,
        "p95PairwiseCosine": float(np.quantile(upper, 0.95)) if upper.size else 1.0,
        "meanAdjacentCosine": float(np.mean(adjacent)) if adjacent.size else 1.0,
        "uniqueArtistCredits": len(
            {normalized_artist(library.artists[index]) for index in selected}
        ),
        "metadataDuplicateExcess": metadata_excess,
        "artistTitleDuplicateExcess": artist_title_excess,
        "nearIdenticalVectorExcess": near_identical_excess,
        "meanGraphIndegree": float(np.mean(graph.indegree[indices])),
        "p95GraphIndegree": float(np.quantile(graph.indegree[indices], 0.95)),
        "top500CoverageAtCosine90": float(np.mean(coverage >= 0.90)),
        "meanTop500BestCoverageCosine": float(np.mean(coverage)),
        "trackIds": [int(library.track_ids[index]) for index in selected],
    }
    if anchor_percentiles is not None:
        chosen = anchor_percentiles[:, indices]
        worst = np.min(chosen, axis=0)
        result.update(
            {
                "meanWorstAnchorPercentile": float(np.mean(worst)),
                "p05WorstAnchorPercentile": float(np.quantile(worst, 0.05)),
                "minimumWorstAnchorPercentile": float(np.min(worst)),
                "meanAnchorPercentiles": [
                    float(value) for value in np.mean(chosen, axis=1)
                ],
            }
        )
    return result


def reciprocal_metrics(
    library: Library,
    seed_index: int,
    selected: Sequence[int],
) -> dict[str, object]:
    indices = np.asarray(selected, dtype=np.int64)
    if indices.size == 0:
        return {}
    started = time.perf_counter()
    ranks = reciprocal_reverse_ranks(library, seed_index, indices)
    return {
        "meanReverseSeedRank": float(np.mean(ranks)),
        "medianReverseSeedRank": float(np.median(ranks)),
        "maximumReverseSeedRank": int(np.max(ranks)),
        "seedInCandidateTop5": int(np.count_nonzero(ranks <= 5)),
        "seedInCandidateTop30": int(np.count_nonzero(ranks <= 30)),
        "measurementMs": (time.perf_counter() - started) * 1000.0,
    }


def queue_record(
    library: Library,
    graph: Graph,
    selected: Sequence[int],
    relevance: np.ndarray,
    positions: np.ndarray,
    elapsed_ms: float,
    anchor_percentiles: np.ndarray | None = None,
    seed_index: int | None = None,
) -> dict[str, object]:
    record = {
        "timingMs": elapsed_ms,
        "metrics": queue_metrics(
            library,
            graph,
            selected,
            relevance,
            positions,
            anchor_percentiles,
        ),
        "orderedResults": [
            {
                **track_label(library, index),
                "objectiveRank": int(positions[index]),
                "objectiveScore": float(relevance[index]),
            }
            for index in selected
        ],
    }
    if seed_index is not None:
        record["reciprocity"] = reciprocal_metrics(library, seed_index, selected)
    record["fingerprint"] = hashlib.sha256(
        np.asarray(
            [library.track_ids[index] for index in selected],
            dtype="<i8",
        ).tobytes()
    ).hexdigest()
    return record


def run_single_seed(
    library: Library,
    graph: Graph,
    case: SeedCase,
) -> dict[str, object]:
    seed = library.index_by_track_id[case.track_id]
    relevance = np.asarray(library.embeddings @ library.embeddings[seed], dtype=np.float32)
    relevance_order = stable_order(relevance, library.track_ids)
    positions = rank_positions(relevance_order)
    excluded = identity_exclusions(library, [seed])
    output: dict[str, object] = {
        "request": (
            "Build a radio from one seed, preserving its particular neighborhood while "
            "testing whether reciprocal rank removes one-way hubs."
        ),
        "seed": {**track_label(library, seed), "slice": case.slice_name},
        "excludedIdentityRows": len(excluded),
        "algorithms": {},
    }

    def add(name: str, formula: str, run: callable) -> None:
        started = time.perf_counter()
        selected = run()
        elapsed = (time.perf_counter() - started) * 1000.0
        repeated = run()
        if list(selected) != list(repeated):
            raise AssertionError(f"{case.key}/{name} is not deterministic")
        output["algorithms"][name] = {
            "formula": formula,
            **queue_record(
                library,
                graph,
                selected,
                relevance,
                positions,
                elapsed,
                seed_index=seed,
            ),
            "repeatExact": True,
        }

    add(
        "closest",
        "complete-domain descending cosine, then track ID; identity and artist constraints",
        lambda: constrained_prefix(
            library, relevance_order, QUEUE_SIZE, excluded
        ),
    )
    pool_count = max(QUEUE_SIZE, int(math.ceil(library.count * 0.02)))
    mmr_ranking = np.asarray(
        [index for index in relevance_order if int(index) not in excluded],
        dtype=np.int64,
    )[:pool_count]
    for lambda_ in (0.5, 0.8, 0.9):
        add(
            f"mmr_{lambda_:g}",
            (
                f"{lambda_:g} * seed cosine - {1-lambda_:g} * maximum selected "
                "overlap, over the exact nearest 2% domain"
            ),
            lambda value=lambda_: select_mmr(
                library,
                relevance,
                mmr_ranking,
                QUEUE_SIZE,
                value,
                excluded,
            ),
        )
    add(
        "dpp_1",
        (
            "full-domain greedy DPP, L(i,j)=q(i)q(j)dot(i,j), "
            "q=max(seed cosine,0)^1"
        ),
        lambda: dpp_sequence(
            library,
            relevance,
            relevance_order,
            QUEUE_SIZE,
            1.0,
            excluded,
        )[0],
    )

    started = time.perf_counter()
    mutual, mutual_evidence = mutual_rankings(
        library,
        seed,
        relevance_order,
        excluded,
        QUEUE_SIZE,
    )
    mutual_total_ms = (time.perf_counter() - started) * 1000.0
    mutual_formulas = {
        "mutual_minimax": (
            "ascending max(forward seed rank, reverse seed rank), then rank sum, then track ID"
        ),
        "mutual_product": (
            "ascending forward seed rank * reverse seed rank, then max rank, then track ID"
        ),
        "mutual_rrf": (
            "descending 1/(60+forward rank)+1/(60+reverse rank), then track ID"
        ),
    }
    for name, selected in mutual.items():
        output["algorithms"][name] = {
            "formula": mutual_formulas[name],
            **queue_record(
                library,
                graph,
                selected,
                relevance,
                positions,
                mutual_total_ms,
                seed_index=seed,
            ),
            "repeatExact": True,
            "mutualEvidence": mutual_evidence,
        }

    # One full greedy sequence makes later deterministic DPP sets unambiguous.
    started = time.perf_counter()
    continuation, gains = dpp_sequence(
        library,
        relevance,
        relevance_order,
        QUEUE_SIZE * 3,
        1.0,
        excluded,
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    blocks: list[dict[str, object]] = []
    for block_index in range(3):
        block = continuation[
            block_index * QUEUE_SIZE : (block_index + 1) * QUEUE_SIZE
        ]
        blocks.append(
            {
                "queueNumber": block_index + 1,
                **queue_record(
                    library,
                    graph,
                    block,
                    relevance,
                    positions,
                    elapsed,
                    seed_index=seed,
                ),
                "firstMarginalGain": (
                    gains[block_index * QUEUE_SIZE]
                    if block_index * QUEUE_SIZE < len(gains)
                    else None
                ),
                "lastMarginalGain": (
                    gains[min(len(gains), (block_index + 1) * QUEUE_SIZE) - 1]
                    if block
                    else None
                ),
            }
        )
    output["anotherQueue"] = {
        "request": (
            "Ask for a fresh deterministic DPP set without hidden listening history."
        ),
        "formula": (
            "queue n is positions (n-1)*30+1 through n*30 of the same complete-domain "
            "greedy DPP sequence"
        ),
        "blocks": blocks,
    }
    return output


def load_text_vector(text_directory: Path, name: str) -> np.ndarray:
    slug = re.sub(r"[^A-Za-z0-9]", "_", name)
    path = text_directory / f"text_emb_{slug}.bin"
    if not path.is_file():
        raise FileNotFoundError(path)
    vector = np.fromfile(path, dtype="<f4")
    if vector.size != EXPECTED_DIM:
        raise ValueError(f"{path} has {vector.size} floats, expected {EXPECTED_DIM}")
    norm = float(np.linalg.norm(vector.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"{path} is non-finite or zero")
    return (vector / np.float32(norm)).astype(np.float32)


def run_multi_anchor(
    library: Library,
    graph: Graph,
    case: MultiAnchorCase,
    text_directory: Path | None,
) -> dict[str, object]:
    song_indices = [library.index_by_track_id[track_id] for track_id in case.track_ids]
    anchors = [
        np.asarray(library.embeddings[index], dtype=np.float32)
        for index in song_indices
    ]
    anchor_labels = [
        f"{library.artists[index]} - {library.titles[index]}" for index in song_indices
    ]
    for text_name in case.text_names:
        if text_directory is None:
            raise ValueError(f"{case.key} needs --text-directory")
        anchors.append(load_text_vector(text_directory, text_name))
        anchor_labels.append(f'text: "{text_name}"')
    started = time.perf_counter()
    objective, percentiles = all_of_objective(library, anchors)
    objective_ms = (time.perf_counter() - started) * 1000.0
    ranking = stable_order(objective, library.track_ids)
    positions = rank_positions(ranking)
    excluded = identity_exclusions(library, song_indices)
    result: dict[str, object] = {
        "request": (
            "Find recordings satisfying every positive anchor; test whether a varied "
            "membership remains a truthful All-of answer."
        ),
        "anchors": anchor_labels,
        "objective": (
            "exp(mean(log(tie-aware active-domain upper-CDF percentile per anchor)))"
        ),
        "objectiveTimingMs": objective_ms,
        "algorithms": {},
    }
    started = time.perf_counter()
    ranked = constrained_prefix(library, ranking, QUEUE_SIZE, excluded)
    elapsed = (time.perf_counter() - started) * 1000.0
    result["algorithms"]["ranked_all_of"] = {
        "formula": "complete-domain descending All-of objective",
        **queue_record(
            library,
            graph,
            ranked,
            objective,
            positions,
            elapsed,
            anchor_percentiles=percentiles,
        ),
    }
    for exponent in (4.0, 16.0, 64.0):
        started = time.perf_counter()
        selected, _ = dpp_sequence(
            library,
            objective,
            ranking,
            QUEUE_SIZE,
            exponent,
            excluded,
        )
        elapsed = (time.perf_counter() - started) * 1000.0
        repeated, _ = dpp_sequence(
            library,
            objective,
            ranking,
            QUEUE_SIZE,
            exponent,
            excluded,
        )
        if selected != repeated:
            raise AssertionError(f"{case.key}/varied_{exponent:g} is not deterministic")
        result["algorithms"][f"varied_all_of_{exponent:g}"] = {
            "formula": (
                "full-domain greedy DPP over All-of quality; "
                f"q=max(All-of objective,0)^{exponent:g}"
            ),
            **queue_record(
                library,
                graph,
                selected,
                objective,
                positions,
                elapsed,
                anchor_percentiles=percentiles,
            ),
            "repeatExact": True,
            "rankedOverlap": len(set(selected) & set(ranked)),
        }
    return result


def run_text_retrieval(
    library: Library,
    graph: Graph,
    query: str,
    text_directory: Path,
) -> dict[str, object]:
    vector = load_text_vector(text_directory, query)
    relevance = np.asarray(library.embeddings @ vector, dtype=np.float32)
    ranking = stable_order(relevance, library.track_ids)
    positions = rank_positions(ranking)
    closest = [int(index) for index in ranking[:QUEUE_SIZE]]
    started = time.perf_counter()
    varied, _ = dpp_sequence(
        library,
        relevance,
        ranking,
        QUEUE_SIZE,
        4.0,
        excluded=set(),
        enforce_artist_constraints=False,
    )
    varied_ms = (time.perf_counter() - started) * 1000.0
    repeated, _ = dpp_sequence(
        library,
        relevance,
        ranking,
        QUEUE_SIZE,
        4.0,
        excluded=set(),
        enforce_artist_constraints=False,
    )
    if varied != repeated:
        raise AssertionError(f"text query {query!r} is not deterministic")
    return {
        "request": f'Retrieve music matching the exact text description "{query}".',
        "query": query,
        "queryVectorSha256": hashlib.sha256(vector.astype("<f4").tobytes()).hexdigest(),
        "algorithms": {
            "closest": {
                "formula": "complete-domain descending text/audio cosine, then track ID",
                **queue_record(
                    library,
                    graph,
                    closest,
                    relevance,
                    positions,
                    0.0,
                ),
                "repeatExact": True,
            },
            "varied_dpp_4": {
                "formula": (
                    "full-domain greedy DPP, L(i,j)=q(i)q(j)dot(i,j), "
                    "q=max(text cosine,0)^4"
                ),
                **queue_record(
                    library,
                    graph,
                    varied,
                    relevance,
                    positions,
                    varied_ms,
                ),
                "rankedOverlap": len(set(varied) & set(closest)),
                "repeatExact": True,
            },
        },
    }


def run_stress(
    library: Library,
    graph: Graph,
    single: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Stress output lengths and candidate domains for the leading challengers."""
    length_records: dict[str, object] = {}
    for case in SEED_CASES:
        seed = library.index_by_track_id[case.track_id]
        relevance = np.asarray(library.embeddings @ library.embeddings[seed], dtype=np.float32)
        order = stable_order(relevance, library.track_ids)
        excluded = identity_exclusions(library, [seed])
        closest_50 = constrained_prefix(library, order, 50, excluded)
        mutual, evidence = mutual_rankings(
            library,
            seed,
            order,
            excluded,
            50,
        )
        mutual_50 = mutual["mutual_minimax"]
        prior_30 = single[case.key]["algorithms"]["mutual_minimax"]["metrics"]["trackIds"]
        length_records[case.key] = {
            "closestCounts": {
                str(count): [int(library.track_ids[index]) for index in closest_50[:count]]
                for count in (10, 30, 50)
            },
            "mutualMinimaxCounts": {
                str(count): [int(library.track_ids[index]) for index in mutual_50[:count]]
                for count in (10, 30, 50)
            },
            "mutual30PrefixStable": (
                [int(library.track_ids[index]) for index in mutual_50[:30]] == prior_30
            ),
            "mutual50Returned": len(mutual_50),
            "mutualCertificate": evidence,
        }

    domains = {
        "receipt_bound_additions_520": np.flatnonzero(library.track_ids > 85_047),
        "deterministic_tenth": np.flatnonzero((library.track_ids % 10) == 0),
    }
    domain_records: dict[str, object] = {}
    for domain_name, domain in domains.items():
        cases: dict[str, object] = {}
        for case in SEED_CASES[:3]:
            seed = library.index_by_track_id[case.track_id]
            relevance = np.asarray(
                library.embeddings @ library.embeddings[seed],
                dtype=np.float32,
            )
            complete_order = stable_order(relevance, library.track_ids)
            complete_positions = rank_positions(complete_order)
            order = stable_order(relevance, library.track_ids, domain)
            excluded = identity_exclusions(library, [seed])
            requested = min(QUEUE_SIZE, int(domain.size))
            closest = constrained_prefix(library, order, requested, excluded)
            mmr = select_mmr(
                library,
                relevance,
                order,
                requested,
                0.8,
                excluded,
            )
            dpp, _ = dpp_sequence(
                library,
                relevance,
                order,
                requested,
                1.0,
                excluded,
            )
            mutual, evidence = mutual_rankings(
                library,
                seed,
                order,
                excluded,
                requested,
            )
            selected_by_name = {
                "closest": closest,
                "mmr_0.8": mmr,
                "dpp_1": dpp,
                "mutual_minimax": mutual["mutual_minimax"],
            }
            domain_set = set(int(index) for index in domain)
            cases[case.key] = {
                "domainCount": int(domain.size),
                "algorithms": {
                    name: {
                        "allRowsInDomain": all(index in domain_set for index in selected),
                        "metrics": queue_metrics(
                            library,
                            graph,
                            selected,
                            relevance,
                            complete_positions,
                        ),
                    }
                    for name, selected in selected_by_name.items()
                },
                "mutualCertificate": evidence,
            }
        domain_records[domain_name] = cases
    return {
        "queueLengths": length_records,
        "candidateDomains": domain_records,
    }


def aggregate_text(text: dict[str, dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("closest", "varied_dpp_4"):
        records = [case["algorithms"][name] for case in text.values()]
        metrics = [record["metrics"] for record in records]
        result[name] = {
            "meanTextCosine": float(
                np.mean([row["meanRelevance"] for row in metrics])
            ),
            "meanP05TextCosine": float(
                np.mean([row["p05Relevance"] for row in metrics])
            ),
            "meanMedianTextRank": float(
                np.mean([row["medianGlobalRank"] for row in metrics])
            ),
            "meanPairwiseCosine": float(
                np.mean([row["meanPairwiseCosine"] for row in metrics])
            ),
            "nearIdenticalVectorExcess": int(
                sum(row["nearIdenticalVectorExcess"] for row in metrics)
            ),
            "artistTitleDuplicateExcess": int(
                sum(row["artistTitleDuplicateExcess"] for row in metrics)
            ),
            "meanRankedOverlap": (
                float(np.mean([record["rankedOverlap"] for record in records]))
                if name == "varied_dpp_4"
                else QUEUE_SIZE
            ),
            "medianPlannerMs": float(
                np.median([record["timingMs"] for record in records])
            ),
            "allRepeatedExactly": all(record["repeatExact"] for record in records),
        }
    return result


def aggregate(
    single: dict[str, dict[str, object]],
    multi: dict[str, dict[str, object]],
) -> dict[str, object]:
    single_names = sorted(
        set.intersection(
            *[
                set(case["algorithms"])
                for case in single.values()
            ]
        )
    )
    single_aggregate: dict[str, object] = {}
    for name in single_names:
        records = [case["algorithms"][name] for case in single.values()]
        metrics = [record["metrics"] for record in records]
        reciprocity = [record["reciprocity"] for record in records]
        single_aggregate[name] = {
            "meanSeedCosine": float(np.mean([row["meanRelevance"] for row in metrics])),
            "meanP05SeedCosine": float(np.mean([row["p05Relevance"] for row in metrics])),
            "meanMedianSeedRank": float(
                np.mean([row["medianGlobalRank"] for row in metrics])
            ),
            "meanPairwiseCosine": float(
                np.mean([row["meanPairwiseCosine"] for row in metrics])
            ),
            "meanUniqueArtists": float(
                np.mean([row["uniqueArtistCredits"] for row in metrics])
            ),
            "meanGraphIndegree": float(
                np.mean([row["meanGraphIndegree"] for row in metrics])
            ),
            "meanReverseSeedRank": float(
                np.mean([row["meanReverseSeedRank"] for row in reciprocity])
            ),
            "meanSeedInCandidateTop30": float(
                np.mean([row["seedInCandidateTop30"] for row in reciprocity])
            ),
            "meanTop500CoverageAtCosine90": float(
                np.mean([row["top500CoverageAtCosine90"] for row in metrics])
            ),
            "medianPlannerMs": float(
                np.median([record["timingMs"] for record in records])
            ),
            "allRepeatedExactly": all(record["repeatExact"] for record in records),
        }
    multi_names = (
        sorted(
            set.intersection(
                *[
                    set(case["algorithms"])
                    for case in multi.values()
                ]
            )
        )
        if multi
        else []
    )
    multi_aggregate: dict[str, object] = {}
    for name in multi_names:
        records = [case["algorithms"][name] for case in multi.values()]
        metrics = [record["metrics"] for record in records]
        multi_aggregate[name] = {
            "meanAllOfObjective": float(
                np.mean([row["meanRelevance"] for row in metrics])
            ),
            "meanWorstAnchorPercentile": float(
                np.mean([row["meanWorstAnchorPercentile"] for row in metrics])
            ),
            "meanP05WorstAnchorPercentile": float(
                np.mean([row["p05WorstAnchorPercentile"] for row in metrics])
            ),
            "meanMedianAllOfRank": float(
                np.mean([row["medianGlobalRank"] for row in metrics])
            ),
            "meanPairwiseCosine": float(
                np.mean([row["meanPairwiseCosine"] for row in metrics])
            ),
            "meanUniqueArtists": float(
                np.mean([row["uniqueArtistCredits"] for row in metrics])
            ),
            "meanRankedOverlap": (
                float(np.mean([record["rankedOverlap"] for record in records]))
                if name != "ranked_all_of"
                else QUEUE_SIZE
            ),
            "medianPlannerMs": float(
                np.median([record["timingMs"] for record in records])
            ),
        }
    continuation: dict[str, object] = {}
    for number in range(3):
        blocks = [
            case["anotherQueue"]["blocks"][number]
            for case in single.values()
        ]
        continuation[str(number + 1)] = {
            "meanSeedCosine": float(
                np.mean([block["metrics"]["meanRelevance"] for block in blocks])
            ),
            "meanP05SeedCosine": float(
                np.mean([block["metrics"]["p05Relevance"] for block in blocks])
            ),
            "meanMedianSeedRank": float(
                np.mean([block["metrics"]["medianGlobalRank"] for block in blocks])
            ),
            "meanPairwiseCosine": float(
                np.mean([block["metrics"]["meanPairwiseCosine"] for block in blocks])
            ),
            "meanUniqueArtists": float(
                np.mean([block["metrics"]["uniqueArtistCredits"] for block in blocks])
            ),
        }
    return {
        "singleSeed": single_aggregate,
        "multiAnchor": multi_aggregate,
        "dppContinuation": continuation,
    }


def qualitative_markdown(
    library: Library,
    single: dict[str, dict[str, object]],
    multi: dict[str, dict[str, object]],
    text: dict[str, dict[str, object]],
) -> str:
    lines = ["# Focused Recommendation Ordered Results", ""]
    for key, case in single.items():
        seed = case["seed"]
        lines.extend(
            [
                f"## Single seed: {seed['artist']} - {seed['title']}",
                "",
                f"Slice: {seed['slice']}.",
                "",
            ]
        )
        for name, record in case["algorithms"].items():
            lines.extend([f"### {name}", ""])
            for position, row in enumerate(record["orderedResults"], start=1):
                lines.append(
                    f"{position}. {row['artist']} - {row['title']} "
                    f"(seed rank #{row['objectiveRank']})"
                )
            lines.append("")
        lines.extend(["### DPP continuation", ""])
        for block in case["anotherQueue"]["blocks"]:
            lines.append(f"Queue {block['queueNumber']}:")
            for position, row in enumerate(block["orderedResults"], start=1):
                lines.append(
                    f"{position}. {row['artist']} - {row['title']} "
                    f"(seed rank #{row['objectiveRank']})"
                )
            lines.append("")
    for key, case in multi.items():
        lines.extend(
            [
                f"## Multi-anchor: {key}",
                "",
                "Anchors: " + " + ".join(case["anchors"]),
                "",
            ]
        )
        for name, record in case["algorithms"].items():
            lines.extend([f"### {name}", ""])
            for position, row in enumerate(record["orderedResults"], start=1):
                lines.append(
                    f"{position}. {row['artist']} - {row['title']} "
                    f"(All-of rank #{row['objectiveRank']})"
                )
            lines.append("")
    for key, case in text.items():
        lines.extend([f"## Text: {case['query']}", ""])
        for name, record in case["algorithms"].items():
            lines.extend([f"### {name}", ""])
            for position, row in enumerate(record["orderedResults"], start=1):
                lines.append(
                    f"{position}. {row['artist']} - {row['title']} "
                    f"(text rank #{row['objectiveRank']})"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--text-directory",
        type=Path,
        help="directory containing exact phone text_emb_<slug>.bin vectors",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="allow a different generation while retaining all shape/order checks",
    )
    parser.add_argument(
        "--single-only",
        action="store_true",
        help="run only the single-seed and DPP-continuation lanes",
    )
    parser.add_argument(
        "--skip-stress",
        action="store_true",
        help="skip queue-length and restricted-domain stress cases",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    database_path = args.snapshot / "poweramp-current-library.db"
    embedding_path = args.snapshot / "poweramp-current-clamp3.emb"
    graph_path = args.snapshot / "poweramp-current-graph.bin"
    hashes = {
        "database": sha256_file(database_path),
        "embeddings": sha256_file(embedding_path),
        "graph": sha256_file(graph_path),
    }
    if not args.skip_hash_check:
        expected = {
            "database": EXPECTED_DATABASE_SHA256,
            "embeddings": EXPECTED_EMBEDDING_SHA256,
            "graph": EXPECTED_GRAPH_SHA256,
        }
        if hashes != expected:
            raise ValueError(f"snapshot hashes changed: expected {expected}, got {hashes}")
    library = load_library(database_path, embedding_path)
    graph = load_graph(graph_path, library)
    started = time.perf_counter()
    single: dict[str, dict[str, object]] = {}
    for position, case in enumerate(SEED_CASES, start=1):
        print(f"single seed {position}/{len(SEED_CASES)}: {case.key}", flush=True)
        single[case.key] = run_single_seed(library, graph, case)
    multi: dict[str, dict[str, object]] = {}
    text: dict[str, dict[str, object]] = {}
    if not args.single_only:
        if args.text_directory is None:
            raise ValueError("multi-anchor and text lanes require --text-directory")
        for position, case in enumerate(MULTI_ANCHOR_CASES, start=1):
            print(f"multi anchor {position}/{len(MULTI_ANCHOR_CASES)}: {case.key}", flush=True)
            multi[case.key] = run_multi_anchor(
                library,
                graph,
                case,
                args.text_directory,
            )
        for position, query in enumerate(TEXT_QUERIES, start=1):
            print(f"text {position}/{len(TEXT_QUERIES)}: {query}", flush=True)
            text[query] = run_text_retrieval(
                library,
                graph,
                query,
                args.text_directory,
            )
    stress = (
        {}
        if args.skip_stress
        else run_stress(library, graph, single)
    )
    summary = aggregate(single, multi)
    if text:
        summary["textRetrieval"] = aggregate_text(text)
    if stress:
        summary["stress"] = {
            "allMutualLengthPrefixesStable": all(
                record["mutual30PrefixStable"]
                for record in stress["queueLengths"].values()
            ),
            "allMutualQueuesReturned50": all(
                record["mutual50Returned"] == 50
                for record in stress["queueLengths"].values()
            ),
            "allRestrictedResultsInPromisedDomain": all(
                algorithm["allRowsInDomain"]
                for domain in stress["candidateDomains"].values()
                for case in domain.values()
                for algorithm in case["algorithms"].values()
            ),
        }
    run = {
        "schemaVersion": 1,
        "snapshot": {
            "path": str(args.snapshot.resolve()),
            "hashes": hashes,
            "trackCount": library.count,
            "embeddingDimension": EXPECTED_DIM,
        },
        "hypotheses": {
            "mutualMatches": (
                "Reciprocal rank may preserve precise single-seed neighborhoods while "
                "reducing one-way hubs more effectively than high-relevance MMR."
            ),
            "variedAllOf": (
                "DPP over the exact All-of objective may provide a broader set without "
                "breaking weakest-anchor satisfaction."
            ),
            "anotherQueue": (
                "Later blocks of one deterministic DPP sequence may provide fresh sets "
                "without hidden history while remaining musically relevant."
            ),
            "textRetrieval": (
                "The existing Closest and Varied text planners should remain distinct on "
                "the complete current generation without acquiring duplicate or domain errors."
            ),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblasThreads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "singleSeed": single,
        "multiAnchor": multi,
        "textRetrieval": text,
        "stress": stress,
        "summary": summary,
        "totalRuntimeSeconds": time.perf_counter() - started,
    }
    run["resultPayloadSha256"] = sha256_json(run)
    atomic_json(args.output / "results.json", run)
    atomic_json(args.output / "summary.json", summary)
    atomic_text(
        args.output / "ordered-results.md",
        qualitative_markdown(library, single, multi, text),
    )
    manifest = {
        "resultsSha256": sha256_file(args.output / "results.json"),
        "summarySha256": sha256_file(args.output / "summary.json"),
        "orderedResultsSha256": sha256_file(args.output / "ordered-results.md"),
        "evaluatorSha256": sha256_file(Path(__file__)),
        "databaseSha256": hashes["database"],
        "embeddingSha256": hashes["embeddings"],
        "graphSha256": hashes["graph"],
    }
    atomic_json(args.output / "run-manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
