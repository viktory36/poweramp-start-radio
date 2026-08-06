#!/usr/bin/env python3
"""Evaluate constrained start-to-destination journeys on the active phone domain.

All route scores use CLaMP3 embeddings only. Artist text is used solely for the explicit
spacing constraint and for human-readable labels; it never contributes to relevance.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import platform
import resource
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import v2_active_composition_eval as active
import v2_queue_eval as queue_eval


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "journey-constrained-2026-07-15"
)

# The five original route probes plus four frozen before inspecting this experiment's paths.
ROUTE_PAIRS: tuple[tuple[int, int], ...] = (
    (13384, 2682),   # Hallucinogen -> Aphex Twin
    (209, 5831),     # Nusrat Fateh Ali Khan -> Burial
    (26499, 2955),   # Seedhe Maut -> Stromae/Pomme
    (3201, 28891),   # Astrix -> Technical Hitch
    (33431, 3201),   # Robin Williamson -> Astrix, deliberate failure probe
    (80437, 42335),  # Bonobo Drift -> Neroche
    (42335, 5831),   # Neroche -> Burial
    (5987, 7838),    # Khruangbin -> D'Gary
    (31830, 28891),  # Tool -> Technical Hitch
)
REQUESTED_LENGTHS: tuple[int, ...] = (12, 20)  # endpoints included

TOP_K_GRAPH = 5
FLEXER_FAR_PERCENT = 95.0
FLEXER_NEAR_FRACTION = 1.0 - FLEXER_FAR_PERCENT / 100.0
INDEPENDENT_POOL = 512
TWO_PART_POOL = 512

# Frozen constrained-search policy. The corridor is 512 exact active-domain nearest
# neighbors per spherical layer, one hundred times denser than the stored K=5 graph.
CANDIDATE_WIDTH = 512
BEAM_WIDTH = 192
BRANCH_WIDTH = 72
MAX_BROAD_DESTINATION_BACKSTEP = 0.02
OUTLIER_ANGLE_DEGREES = 35.0
EDGE_WEIGHT = 1.0
CORRIDOR_WEIGHT = 1.0
CROSS_TRACK_WEIGHT = 0.35
BACKWARD_WEIGHT = 10.0
OUTLIER_WEIGHT = 4.0
MAX_EDGE_WEIGHT = 0.5
MAX_CORRIDOR_WEIGHT = 0.5

MAX_PER_ARTIST = queue_eval.DEFAULT_MAX_PER_ARTIST
MIN_ARTIST_SPACING = queue_eval.DEFAULT_MIN_ARTIST_SPACING


@dataclass(frozen=True)
class JourneyLibrary:
    track_ids: np.ndarray
    embeddings: np.ndarray
    artists: tuple[str | None, ...]
    albums: tuple[str | None, ...]
    titles: tuple[str | None, ...]
    durations_ms: np.ndarray
    file_paths: tuple[str, ...]

    @property
    def count(self) -> int:
        return int(self.track_ids.size)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class BeamState:
    path: tuple[int, ...]
    cumulative_cost: float
    max_edge_angle: float
    max_target_error: float

    def objective_cost(self) -> float:
        return (
            self.cumulative_cost
            + MAX_EDGE_WEIGHT * self.max_edge_angle * self.max_edge_angle
            + MAX_CORRIDOR_WEIGHT * self.max_target_error * self.max_target_error
        )

    def priority(self, library: JourneyLibrary) -> tuple[float, float, float, tuple[int, ...]]:
        return (
            self.objective_cost(),
            self.max_edge_angle,
            self.max_target_error,
            tuple(int(library.track_ids[index]) for index in self.path),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def normalized_artist(value: str | None) -> str | None:
    return value.lower() if value is not None else None


def artist_valid(path: Sequence[int], library: JourneyLibrary) -> bool:
    for position, candidate in enumerate(path):
        artist = normalized_artist(library.artists[candidate])
        if artist is None:
            continue
        previous = path[:position]
        count = sum(
            normalized_artist(library.artists[index]) == artist for index in previous
        )
        if count >= MAX_PER_ARTIST:
            return False
        if MIN_ARTIST_SPACING > 0 and any(
            normalized_artist(library.artists[index]) == artist
            for index in previous[-MIN_ARTIST_SPACING:]
        ):
            return False
    return True


def can_append_artist(candidate: int, path: Sequence[int], library: JourneyLibrary) -> bool:
    artist = normalized_artist(library.artists[candidate])
    if artist is None:
        return True
    if sum(normalized_artist(library.artists[index]) == artist for index in path) >= MAX_PER_ARTIST:
        return False
    return not any(
        normalized_artist(library.artists[index]) == artist
        for index in path[-MIN_ARTIST_SPACING:]
    )


def reserve_destination_artist(
    candidate: int,
    layer: int,
    total_length: int,
    destination: int,
    library: JourneyLibrary,
) -> bool:
    if layer < total_length - 1 - MIN_ARTIST_SPACING:
        return True
    destination_artist = normalized_artist(library.artists[destination])
    return destination_artist is None or normalized_artist(library.artists[candidate]) != destination_artist


def parse_graph_blob(db_path: Path, full: queue_eval.Library) -> np.ndarray:
    with queue_eval.read_only_connection(db_path.resolve()) as connection:
        row = connection.execute(
            "SELECT data FROM binary_data WHERE key = 'knn_graph'"
        ).fetchone()
    if row is None:
        raise ValueError("database has no knn_graph")
    blob = memoryview(row[0])
    count, k = struct.unpack_from("<II", blob, 0)
    if count != full.count or k != TOP_K_GRAPH:
        raise ValueError(f"unexpected stored graph shape {count} x {k}")
    ids = np.frombuffer(blob, dtype="<i8", count=count, offset=8)
    if not np.array_equal(ids, full.track_ids):
        raise ValueError("stored graph ID map does not match the database")
    entry_dtype = np.dtype([("neighbor", "<u4"), ("weight", "<f4")])
    offset = 8 + count * 8
    entries = np.frombuffer(
        blob, dtype=entry_dtype, count=count * k, offset=offset
    ).reshape(count, k)
    neighbors = entries["neighbor"].astype(np.int64)
    if np.any(neighbors < 0) or np.any(neighbors >= count):
        raise ValueError("stored graph contains an invalid neighbor")
    return neighbors


def load_active_library_and_graph(
    db_path: Path,
    active_catalog_path: Path,
    verify_hash: bool,
) -> tuple[JourneyLibrary, list[dict[int, float]], str, dict[str, object]]:
    catalog = active.parse_active_catalog(active_catalog_path)
    full, db_hash = queue_eval.load_library(db_path, verify_hash=verify_hash)
    all_partition = np.sort(
        np.concatenate((catalog.active_track_ids, catalog.quarantined_track_ids))
    )
    if not np.array_equal(full.track_ids, all_partition):
        raise ValueError("active catalog does not partition the database")
    source = np.searchsorted(full.track_ids, catalog.active_track_ids)
    if not np.array_equal(full.track_ids[source], catalog.active_track_ids):
        raise ValueError("active projection is not one-to-one")
    library = JourneyLibrary(
        track_ids=full.track_ids[source].copy(),
        embeddings=full.embeddings[source].copy(),
        artists=tuple(full.artists[int(index)] for index in source),
        albums=tuple(full.albums[int(index)] for index in source),
        titles=tuple(full.titles[int(index)] for index in source),
        durations_ms=full.durations_ms[source].copy(),
        file_paths=tuple(full.file_paths[int(index)] for index in source),
    )
    if library.count != active.EXPECTED_ACTIVE_COUNT or library.dim != queue_eval.EXPECTED_DIM:
        raise ValueError("active journey library has the wrong shape")

    full_neighbors = parse_graph_blob(db_path, full)
    full_to_active = np.full(full.count, -1, dtype=np.int64)
    full_to_active[source] = np.arange(library.count, dtype=np.int64)
    adjacency: list[dict[int, float]] = [dict() for _ in range(library.count)]
    active_edge_count = 0
    for active_source, full_source in enumerate(source):
        for full_target in full_neighbors[int(full_source)]:
            active_target = int(full_to_active[int(full_target)])
            if active_target < 0 or active_target == active_source:
                continue
            cosine = float(
                np.dot(library.embeddings[active_source], library.embeddings[active_target])
            )
            cost = math.acos(float(np.clip(cosine, -1.0, 1.0)))
            old = adjacency[active_source].get(active_target)
            if old is None or cost < old:
                adjacency[active_source][active_target] = cost
                adjacency[active_target][active_source] = cost
            active_edge_count += 1
    graph_info = {
        "stored_k": TOP_K_GRAPH,
        "directed_active_edges_before_undirected_union": active_edge_count,
        "undirected_edge_count": sum(len(row) for row in adjacency) // 2,
    }
    return library, adjacency, db_hash, graph_info


def track_summary(library: JourneyLibrary, index: int) -> dict[str, object]:
    return {
        "index": int(index),
        "track_id": int(library.track_ids[index]),
        "artist": library.artists[index],
        "album": library.albums[index],
        "title": library.titles[index],
        "duration_ms": int(library.durations_ms[index]),
    }


def rank_indices(
    scores: np.ndarray,
    library: JourneyLibrary,
    count: int,
    excluded: set[int] | None = None,
) -> np.ndarray:
    eligible = np.ones(library.count, dtype=bool)
    if excluded:
        eligible[np.fromiter(sorted(excluded), dtype=np.int64)] = False
    indices = np.flatnonzero(eligible)
    # Full stable ordering makes the boundary exact even when duplicate/copy rows
    # have equal Float32 scores. Candidate construction is an experiment input, so
    # an argpartition boundary with unspecified equal-score membership is not enough.
    order = np.lexsort((library.track_ids[indices], -scores[indices]))
    return indices[order[:count]]


def slerp(start: np.ndarray, destination: np.ndarray, fraction: float) -> np.ndarray:
    start64 = np.asarray(start, dtype=np.float64)
    destination64 = np.asarray(destination, dtype=np.float64)
    start64 /= np.linalg.norm(start64)
    destination64 /= np.linalg.norm(destination64)
    cosine = float(np.clip(np.dot(start64, destination64), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        result = (1.0 - fraction) * start64 + fraction * destination64
    else:
        sine = math.sin(angle)
        result = (
            math.sin((1.0 - fraction) * angle) / sine * start64
            + math.sin(fraction * angle) / sine * destination64
        )
    result /= np.linalg.norm(result)
    return result.astype(np.float32)


def corridor_geometry(
    embeddings: np.ndarray,
    start_embedding: np.ndarray,
    destination_embedding: np.ndarray,
    fractions: np.ndarray,
) -> dict[str, np.ndarray | float]:
    start64 = np.asarray(start_embedding, dtype=np.float64)
    destination64 = np.asarray(destination_embedding, dtype=np.float64)
    start64 /= np.linalg.norm(start64)
    destination64 /= np.linalg.norm(destination64)
    endpoint_cosine = float(np.clip(np.dot(start64, destination64), -1.0, 1.0))
    endpoint_angle = math.acos(endpoint_cosine)
    if endpoint_angle < 1e-8:
        raise ValueError("journey endpoints are not directionally distinct")
    tangent = destination64 - endpoint_cosine * start64
    tangent /= np.linalg.norm(tangent)
    values = np.asarray(embeddings, dtype=np.float64)
    start_coordinate = values @ start64
    tangent_coordinate = values @ tangent
    route_progress = np.arctan2(tangent_coordinate, start_coordinate) / endpoint_angle
    plane_magnitude = np.sqrt(start_coordinate * start_coordinate + tangent_coordinate * tangent_coordinate)
    cross_track = np.arccos(np.clip(plane_magnitude, 0.0, 1.0))
    targets = np.stack(
        [slerp(start64, destination64, float(fraction)) for fraction in fractions], axis=0
    ).astype(np.float64)
    target_cosines = np.sum(values * targets, axis=1)
    target_errors = np.arccos(np.clip(target_cosines, -1.0, 1.0))
    destination_cosines = values @ destination64
    start_cosines = values @ start64
    return {
        "endpoint_angle": endpoint_angle,
        "route_progress": route_progress,
        "cross_track": cross_track,
        "target_errors": target_errors,
        "target_cosines": target_cosines,
        "destination_cosines": destination_cosines,
        "start_cosines": start_cosines,
    }


def shortest_path(
    adjacency: Sequence[dict[int, float]], start: int, destination: int
) -> list[int] | None:
    distance = {start: 0.0}
    previous: dict[int, int] = {}
    queue: list[tuple[float, int]] = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance.get(node):
            continue
        if node == destination:
            path = [node]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            return path
        for neighbor, edge_cost in adjacency[node].items():
            candidate = cost + edge_cost
            old = distance.get(neighbor)
            if old is None or candidate < old - 1e-15:
                distance[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return None


def independent_slerp_path(
    library: JourneyLibrary,
    start: int,
    destination: int,
    length: int,
) -> list[int]:
    selected = [start]
    excluded = {start, destination}
    for layer in range(1, length - 1):
        query = slerp(
            library.embeddings[start], library.embeddings[destination], layer / (length - 1)
        )
        scores = np.asarray(library.embeddings @ query, dtype=np.float32)
        candidates = rank_indices(scores, library, INDEPENDENT_POOL, excluded | set(selected))
        chosen = next(
            (
                int(candidate)
                for candidate in candidates
                if can_append_artist(int(candidate), selected, library)
                and reserve_destination_artist(
                    int(candidate), layer, length, destination, library
                )
            ),
            None,
        )
        if chosen is None:
            raise RuntimeError("independent slerp exhausted its candidate pool")
        selected.append(chosen)
    if not can_append_artist(destination, selected, library):
        raise RuntimeError("independent slerp failed to reserve the destination artist")
    selected.append(destination)
    return selected


def flexer_ratio_path(
    library: JourneyLibrary,
    start: int,
    destination: int,
    length: int,
) -> tuple[list[int], dict[str, object]]:
    """Strong endpoint-stable adaptation of Flexer et al.'s divergence-ratio list."""
    start_cosine = np.asarray(library.embeddings @ library.embeddings[start], dtype=np.float32)
    destination_cosine = np.asarray(
        library.embeddings @ library.embeddings[destination], dtype=np.float32
    )
    start_divergence = np.arccos(np.clip(start_cosine, -1.0, 1.0)).astype(np.float64)
    destination_divergence = np.arccos(
        np.clip(destination_cosine, -1.0, 1.0)
    ).astype(np.float64)
    denominator = start_divergence + destination_divergence
    ratio = np.divide(
        start_divergence,
        denominator,
        out=np.full(library.count, 0.5, dtype=np.float64),
        where=denominator > 1e-12,
    )
    near_count = max(1, int(math.ceil(FLEXER_NEAR_FRACTION * library.count)))
    start_near = set(rank_indices(-start_divergence, library, near_count).tolist())
    destination_near = set(
        rank_indices(-destination_divergence, library, near_count).tolist()
    )
    corridor = np.asarray(sorted(start_near | destination_near), dtype=np.int64)
    corridor = corridor[(corridor != start) & (corridor != destination)]
    selected = [start]
    for layer in range(1, length - 1):
        ideal = layer / (length - 1)
        available = corridor[~np.isin(corridor, np.asarray(selected, dtype=np.int64))]
        order = np.lexsort(
            (
                library.track_ids[available],
                denominator[available],
                np.abs(ratio[available] - ideal),
            )
        )
        chosen = next(
            (
                int(available[position])
                for position in order
                if can_append_artist(int(available[position]), selected, library)
                and reserve_destination_artist(
                    int(available[position]), layer, length, destination, library
                )
            ),
            None,
        )
        if chosen is None:
            raise RuntimeError("Flexer ratio baseline exhausted its retained corridor")
        selected.append(chosen)
    if not can_append_artist(destination, selected, library):
        raise RuntimeError("Flexer ratio baseline failed to reserve destination artist")
    selected.append(destination)
    return selected, {
        "divergence": "angular distance arccos(cosine)",
        "bounded_ratio": "d(start) / (d(start) + d(destination))",
        "retained_candidate_count": int(corridor.size),
        "near_fraction_per_endpoint": FLEXER_NEAR_FRACTION,
        "paper_far_percent": FLEXER_FAR_PERCENT,
        "adaptation_note": (
            "The paper's approximate Gaussian divergence has nonzero self-distance. "
            "The bounded ratio preserves ratio ordering while giving cosine distance "
            "exact 0/1 endpoints without an arbitrary epsilon."
        ),
    }


def build_right_half(
    library: JourneyLibrary,
    start: int,
    destination: int,
    count: int,
) -> list[int]:
    destination_scores = np.asarray(
        library.embeddings @ library.embeddings[destination], dtype=np.float32
    )
    ranked = rank_indices(destination_scores, library, TWO_PART_POOL, {start, destination})
    backwards = [destination]
    for candidate in ranked:
        candidate_int = int(candidate)
        if can_append_artist(candidate_int, backwards, library):
            backwards.append(candidate_int)
        if len(backwards) == count + 1:
            break
    if len(backwards) != count + 1:
        raise RuntimeError("two-part destination half exhausted")
    return list(reversed(backwards[1:]))


def two_part_path(
    library: JourneyLibrary,
    start: int,
    destination: int,
    length: int,
) -> list[int]:
    intermediate_count = length - 2
    left_count = intermediate_count // 2
    right_count = intermediate_count - left_count
    right = build_right_half(library, start, destination, right_count)
    start_scores = np.asarray(library.embeddings @ library.embeddings[start], dtype=np.float32)
    ranked = rank_indices(start_scores, library, TWO_PART_POOL, {start, destination, *right})
    states: list[tuple[float, tuple[int, ...]]] = [(0.0, (start,))]
    for _ in range(left_count):
        expanded: list[tuple[float, tuple[int, ...]]] = []
        for score, path in states:
            for candidate in ranked[:96]:
                candidate_int = int(candidate)
                if candidate_int in path or not can_append_artist(candidate_int, path, library):
                    continue
                expanded.append(
                    (score - float(start_scores[candidate_int]), path + (candidate_int,))
                )
        expanded.sort(
            key=lambda item: (
                item[0], tuple(int(library.track_ids[index]) for index in item[1])
            )
        )
        states = expanded[:128]
        if not states:
            raise RuntimeError("two-part start half exhausted")
    for _, left in states:
        full = list(left) + right + [destination]
        if len(set(full)) == len(full) and artist_valid(full, library):
            return full
    raise RuntimeError("two-part baseline could not satisfy the boundary artist constraint")


def exact_layer_candidates(
    library: JourneyLibrary,
    start: int,
    destination: int,
    length: int,
) -> list[np.ndarray]:
    layers: list[np.ndarray] = [np.asarray([start], dtype=np.int64)]
    excluded = {start, destination}
    for layer in range(1, length - 1):
        query = slerp(
            library.embeddings[start], library.embeddings[destination], layer / (length - 1)
        )
        scores = np.asarray(library.embeddings @ query, dtype=np.float32)
        layers.append(rank_indices(scores, library, CANDIDATE_WIDTH, excluded))
    layers.append(np.asarray([destination], dtype=np.int64))
    return layers


def constrained_beam_path(
    library: JourneyLibrary,
    start: int,
    destination: int,
    length: int,
) -> tuple[list[int], dict[str, object]]:
    layers = exact_layer_candidates(library, start, destination, length)
    fractions = np.arange(length, dtype=np.float64) / (length - 1)
    start_embedding = library.embeddings[start]
    destination_embedding = library.embeddings[destination]
    states = [BeamState((start,), 0.0, 0.0, 0.0)]
    layer_stats: list[dict[str, object]] = []
    outlier_threshold = math.radians(OUTLIER_ANGLE_DEGREES)

    for layer in range(1, length):
        candidates = layers[layer]
        candidate_fractions = np.full(candidates.size, fractions[layer], dtype=np.float64)
        geometry = corridor_geometry(
            library.embeddings[candidates],
            start_embedding,
            destination_embedding,
            candidate_fractions,
        )
        destination_cosines = np.asarray(geometry["destination_cosines"])
        target_errors = np.asarray(geometry["target_errors"])
        cross_track = np.asarray(geometry["cross_track"])
        last = np.asarray([state.path[-1] for state in states], dtype=np.int64)
        transition_cosines = np.asarray(
            library.embeddings[last] @ library.embeddings[candidates].T,
            dtype=np.float32,
        )
        transition_angles = np.arccos(np.clip(transition_cosines, -1.0, 1.0))
        previous_destination = np.asarray(
            library.embeddings[last] @ destination_embedding, dtype=np.float64
        )

        expanded: list[BeamState] = []
        considered = 0
        artist_rejected = 0
        progress_rejected = 0
        identity_rejected = 0
        for state_index, state in enumerate(states):
            backward = np.maximum(
                0.0, previous_destination[state_index] - destination_cosines
            )
            valid_progress = backward <= MAX_BROAD_DESTINATION_BACKSTEP + 1e-12
            edge_angles = transition_angles[state_index].astype(np.float64)
            offroute = np.maximum(0.0, target_errors - outlier_threshold)
            local_cost = (
                EDGE_WEIGHT * edge_angles * edge_angles
                + CORRIDOR_WEIGHT * target_errors * target_errors
                + CROSS_TRACK_WEIGHT * cross_track * cross_track
                + BACKWARD_WEIGHT * backward * backward
                + OUTLIER_WEIGHT * offroute * offroute
            )
            order = np.lexsort(
                (library.track_ids[candidates], local_cost)
            )
            branched = 0
            for candidate_position in order:
                considered += 1
                if not valid_progress[candidate_position]:
                    progress_rejected += 1
                    continue
                candidate = int(candidates[candidate_position])
                if candidate in state.path:
                    identity_rejected += 1
                    continue
                if not can_append_artist(candidate, state.path, library):
                    artist_rejected += 1
                    continue
                if layer < length - 1 and not reserve_destination_artist(
                    candidate, layer, length, destination, library
                ):
                    artist_rejected += 1
                    continue
                expanded.append(
                    BeamState(
                        path=state.path + (candidate,),
                        cumulative_cost=state.cumulative_cost
                        + float(local_cost[candidate_position]),
                        max_edge_angle=max(
                            state.max_edge_angle, float(edge_angles[candidate_position])
                        ),
                        max_target_error=max(
                            state.max_target_error,
                            float(target_errors[candidate_position]),
                        ),
                    )
                )
                branched += 1
                if branched >= BRANCH_WIDTH:
                    break
        if not expanded:
            raise RuntimeError(
                f"constrained beam exhausted at layer {layer}/{length - 1}"
            )
        expanded.sort(key=lambda state: state.priority(library))
        states = expanded[:BEAM_WIDTH]
        layer_stats.append(
            {
                "layer": layer,
                "candidate_count": int(candidates.size),
                "incoming_states": int(last.size),
                "expanded_states": len(expanded),
                "retained_states": len(states),
                "considered_transitions": considered,
                "artist_rejected": artist_rejected,
                "progress_rejected": progress_rejected,
                "identity_rejected": identity_rejected,
            }
        )
    winner = states[0]
    path = list(winner.path)
    if len(path) != length or path[0] != start or path[-1] != destination:
        raise AssertionError("constrained search did not lock length and endpoints")
    if len(path) != len(set(path)) or not artist_valid(path, library):
        raise AssertionError("constrained search violated identity or artist state")
    return path, {
        "cumulative_cost": winner.cumulative_cost,
        "search_joint_objective_cost": winner.objective_cost(),
        "max_search_edge_angle_degrees": math.degrees(winner.max_edge_angle),
        "max_search_target_error_degrees": math.degrees(winner.max_target_error),
        "layer_stats": layer_stats,
    }


def duplicate_diagnostics(path: Sequence[int], library: JourneyLibrary) -> dict[str, object]:
    exact_embedding_pairs: list[dict[str, object]] = []
    near_embedding_pairs: list[dict[str, object]] = []
    for first_position in range(len(path)):
        for second_position in range(first_position + 1, len(path)):
            cosine = float(
                np.dot(
                    library.embeddings[path[first_position]],
                    library.embeddings[path[second_position]],
                )
            )
            record = {
                "first_position": first_position,
                "second_position": second_position,
                "first_track_id": int(library.track_ids[path[first_position]]),
                "second_track_id": int(library.track_ids[path[second_position]]),
                "embedding_cosine": cosine,
            }
            if cosine >= 1.0 - 1e-7:
                exact_embedding_pairs.append(record)
            elif cosine >= 0.995:
                near_embedding_pairs.append(record)
    return {
        "repeated_active_track_ids": len(path) - len(set(path)),
        "exact_embedding_pairs": exact_embedding_pairs,
        "near_embedding_pairs_ge_0_995": near_embedding_pairs,
        "identity_scope": (
            "Only repeated active track IDs are proven identity in this frozen legacy "
            "projection. Similar embeddings are diagnostic and are not collapsed."
        ),
    }


def path_metrics(
    path: Sequence[int],
    library: JourneyLibrary,
    requested_start: int,
    requested_destination: int,
    requested_length: int | None,
) -> dict[str, object]:
    embeddings = library.embeddings[np.asarray(path, dtype=np.int64)]
    adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    adjacent_angles = np.arccos(np.clip(adjacent.astype(np.float64), -1.0, 1.0))
    fractions = np.arange(len(path), dtype=np.float64) / (len(path) - 1)
    geometry = corridor_geometry(
        embeddings,
        library.embeddings[requested_start],
        library.embeddings[requested_destination],
        fractions,
    )
    destination_progress = np.asarray(geometry["destination_cosines"], dtype=np.float64)
    start_progress = np.asarray(geometry["start_cosines"], dtype=np.float64)
    route_progress = np.asarray(geometry["route_progress"], dtype=np.float64)
    target_errors = np.asarray(geometry["target_errors"], dtype=np.float64)
    cross_track = np.asarray(geometry["cross_track"], dtype=np.float64)
    destination_diffs = np.diff(destination_progress)
    route_diffs = np.diff(route_progress)
    interior_errors = target_errors[1:-1]
    interior_cross = cross_track[1:-1]
    backward = np.maximum(0.0, -destination_diffs)
    offroute = np.maximum(
        0.0, target_errors[1:] - math.radians(OUTLIER_ANGLE_DEGREES)
    )
    joint_step_costs = (
        EDGE_WEIGHT * adjacent_angles * adjacent_angles
        + CORRIDOR_WEIGHT * target_errors[1:] * target_errors[1:]
        + CROSS_TRACK_WEIGHT * cross_track[1:] * cross_track[1:]
        + BACKWARD_WEIGHT * backward * backward
        + OUTLIER_WEIGHT * offroute * offroute
    )
    joint_cost = (
        float(np.sum(joint_step_costs))
        + MAX_EDGE_WEIGHT * float(np.max(adjacent_angles)) ** 2
        + MAX_CORRIDOR_WEIGHT * float(np.max(target_errors)) ** 2
    )
    return {
        "actual_length": len(path),
        "requested_length": requested_length,
        "length_exact": requested_length is None or len(path) == requested_length,
        "start_endpoint_exact": path[0] == requested_start,
        "destination_endpoint_exact": path[-1] == requested_destination,
        "mean_adjacent_cosine": float(np.mean(adjacent)),
        "minimum_adjacent_cosine": float(np.min(adjacent)),
        "adjacent_cosines": adjacent.tolist(),
        "joint_objective_cost": joint_cost,
        "joint_step_costs": joint_step_costs.tolist(),
        "destination_progress": destination_progress.tolist(),
        "destination_monotonic_violations": int(np.sum(destination_diffs < -1e-6)),
        "destination_broad_violations": int(
            np.sum(destination_diffs < -MAX_BROAD_DESTINATION_BACKSTEP - 1e-9)
        ),
        "maximum_destination_backstep": float(
            max(0.0, -float(np.min(destination_diffs)))
        ),
        "start_monotonic_violations": int(np.sum(np.diff(start_progress) > 1e-6)),
        "route_progress": route_progress.tolist(),
        "route_progress_monotonic_violations": int(np.sum(route_diffs < -1e-6)),
        "mean_absolute_route_progress_error": float(
            np.mean(np.abs(route_progress - fractions))
        ),
        "maximum_absolute_route_progress_error": float(
            np.max(np.abs(route_progress - fractions))
        ),
        "mean_interpolation_angle_error_degrees": float(
            math.degrees(float(np.mean(interior_errors))) if interior_errors.size else 0.0
        ),
        "maximum_interpolation_angle_error_degrees": float(
            math.degrees(float(np.max(interior_errors))) if interior_errors.size else 0.0
        ),
        "mean_cross_track_angle_degrees": float(
            math.degrees(float(np.mean(interior_cross))) if interior_cross.size else 0.0
        ),
        "maximum_cross_track_angle_degrees": float(
            math.degrees(float(np.max(interior_cross))) if interior_cross.size else 0.0
        ),
        "outlier_count_over_35_degrees": int(
            np.sum(interior_errors > math.radians(OUTLIER_ANGLE_DEGREES))
        ),
        "artist_constraints_valid": artist_valid(path, library),
        "duplicates": duplicate_diagnostics(path, library),
    }


def method_record(
    name: str,
    path: Sequence[int],
    library: JourneyLibrary,
    start: int,
    destination: int,
    requested_length: int | None,
    details: dict[str, object] | None = None,
) -> dict[str, object]:
    metrics = path_metrics(path, library, start, destination, requested_length)
    if details and "search_joint_objective_cost" in details:
        search_cost = float(details["search_joint_objective_cost"])
        measured_cost = float(metrics["joint_objective_cost"])
        if not math.isclose(search_cost, measured_cost, rel_tol=1e-6, abs_tol=1e-7):
            raise AssertionError(
                "constrained search and reported joint objective disagree: "
                f"search={search_cost:.12f}, measured={measured_cost:.12f}"
            )
    record = {
        "method": name,
        "metrics": metrics,
        "track_ids": [int(library.track_ids[index]) for index in path],
        "tracks": [track_summary(library, index) for index in path],
    }
    if details:
        record["details"] = details
    return record


def dominance(
    candidate: dict[str, object], baseline: dict[str, object]
) -> dict[str, object]:
    new = candidate["metrics"]
    old = baseline["metrics"]
    checks = {
        "endpoint_exact": bool(new["start_endpoint_exact"] and new["destination_endpoint_exact"]),
        "requested_length_exact": bool(new["length_exact"]),
        "artist_valid": bool(new["artist_constraints_valid"]),
        "broad_progress_not_worse": int(new["destination_broad_violations"])
        <= int(old["destination_broad_violations"]),
        "mean_adjacency_not_worse": float(new["mean_adjacent_cosine"])
        >= float(old["mean_adjacent_cosine"]) - 1e-7,
        "minimum_adjacency_not_worse": float(new["minimum_adjacent_cosine"])
        >= float(old["minimum_adjacent_cosine"]) - 1e-7,
        "mean_corridor_error_not_worse": float(new["mean_interpolation_angle_error_degrees"])
        <= float(old["mean_interpolation_angle_error_degrees"]) + 1e-7,
        "maximum_corridor_error_not_worse": float(new["maximum_interpolation_angle_error_degrees"])
        <= float(old["maximum_interpolation_angle_error_degrees"]) + 1e-7,
        "outliers_not_worse": int(new["outlier_count_over_35_degrees"])
        <= int(old["outlier_count_over_35_degrees"]),
    }
    strict = {
        "fewer_broad_progress_violations": int(new["destination_broad_violations"])
        < int(old["destination_broad_violations"]),
        "better_mean_adjacency": float(new["mean_adjacent_cosine"])
        > float(old["mean_adjacent_cosine"]) + 1e-7,
        "better_minimum_adjacency": float(new["minimum_adjacent_cosine"])
        > float(old["minimum_adjacent_cosine"]) + 1e-7,
        "better_mean_corridor_error": float(new["mean_interpolation_angle_error_degrees"])
        < float(old["mean_interpolation_angle_error_degrees"]) - 1e-7,
        "fewer_outliers": int(new["outlier_count_over_35_degrees"])
        < int(old["outlier_count_over_35_degrees"]),
    }
    return {
        "checks": checks,
        "strict_improvements": strict,
        "componentwise_dominates": all(checks.values()) and any(strict.values()),
        "joint_gate_checks": {
            "endpoint_exact": checks["endpoint_exact"],
            "requested_length_exact": checks["requested_length_exact"],
            "artist_valid": checks["artist_valid"],
            "broad_progress_not_worse": checks["broad_progress_not_worse"],
            "joint_objective_better": float(new["joint_objective_cost"])
            < float(old["joint_objective_cost"]) - 1e-9,
        },
        "joint_objective_delta_candidate_minus_baseline": float(
            new["joint_objective_cost"]
        )
        - float(old["joint_objective_cost"]),
    }


def aggregate(records: Sequence[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, int | None], list[dict[str, object]]] = {}
    dominance_rows: list[dict[str, object]] = []
    for pair in records:
        graph = pair.get("graph_shortest")
        for length_record in pair["lengths"]:
            length = int(length_record["requested_length"])
            methods = length_record["methods"]
            for name, record in methods.items():
                grouped.setdefault((name, length), []).append(record)
            constrained = methods["constrained_beam"]
            baselines = {
                name: record for name, record in methods.items() if name != "constrained_beam"
            }
            if graph is not None:
                baselines["stored_k5_graph_shortest"] = graph
            for baseline_name, baseline in baselines.items():
                dominance_rows.append(
                    row := {
                        "pair_id": pair["id"],
                        "requested_length": length,
                        "baseline": baseline_name,
                        **dominance(constrained, baseline),
                    }
                )
                row["beats_joint_gate"] = all(row["joint_gate_checks"].values())

    summaries: dict[str, object] = {}
    for (name, length), methods in sorted(grouped.items()):
        metrics = [record["metrics"] for record in methods]
        key = f"{name}/length_{length}"
        summaries[key] = {
            "pairs": len(metrics),
            "endpoint_exact_count": sum(
                bool(item["start_endpoint_exact"] and item["destination_endpoint_exact"])
                for item in metrics
            ),
            "length_exact_count": sum(bool(item["length_exact"]) for item in metrics),
            "artist_valid_count": sum(bool(item["artist_constraints_valid"]) for item in metrics),
            "total_destination_violations": sum(
                int(item["destination_monotonic_violations"]) for item in metrics
            ),
            "total_broad_destination_violations": sum(
                int(item["destination_broad_violations"]) for item in metrics
            ),
            "mean_adjacent_cosine": float(
                np.mean([item["mean_adjacent_cosine"] for item in metrics])
            ),
            "mean_minimum_adjacent_cosine": float(
                np.mean([item["minimum_adjacent_cosine"] for item in metrics])
            ),
            "worst_minimum_adjacent_cosine": float(
                np.min([item["minimum_adjacent_cosine"] for item in metrics])
            ),
            "mean_interpolation_error_degrees": float(
                np.mean([item["mean_interpolation_angle_error_degrees"] for item in metrics])
            ),
            "mean_maximum_interpolation_error_degrees": float(
                np.mean([item["maximum_interpolation_angle_error_degrees"] for item in metrics])
            ),
            "total_outliers": sum(
                int(item["outlier_count_over_35_degrees"]) for item in metrics
            ),
            "total_repeated_track_ids": sum(
                int(item["duplicates"]["repeated_active_track_ids"]) for item in metrics
            ),
            "mean_joint_objective_cost": float(
                np.mean([item["joint_objective_cost"] for item in metrics])
            ),
        }
    graph_methods = [record["graph_shortest"] for record in records if record["graph_shortest"]]
    if graph_methods:
        metrics = [record["metrics"] for record in graph_methods]
        summaries["stored_k5_graph_shortest/variable_length"] = {
            "pairs": len(metrics),
            "endpoint_exact_count": sum(
                bool(item["start_endpoint_exact"] and item["destination_endpoint_exact"])
                for item in metrics
            ),
            "artist_valid_count": sum(bool(item["artist_constraints_valid"]) for item in metrics),
            "mean_actual_length": float(np.mean([item["actual_length"] for item in metrics])),
            "total_broad_destination_violations": sum(
                int(item["destination_broad_violations"]) for item in metrics
            ),
            "mean_adjacent_cosine": float(
                np.mean([item["mean_adjacent_cosine"] for item in metrics])
            ),
            "mean_minimum_adjacent_cosine": float(
                np.mean([item["minimum_adjacent_cosine"] for item in metrics])
            ),
            "worst_minimum_adjacent_cosine": float(
                np.min([item["minimum_adjacent_cosine"] for item in metrics])
            ),
            "mean_interpolation_error_degrees": float(
                np.mean([item["mean_interpolation_angle_error_degrees"] for item in metrics])
            ),
            "total_outliers": sum(
                int(item["outlier_count_over_35_degrees"]) for item in metrics
            ),
            "mean_joint_objective_cost": float(
                np.mean([item["joint_objective_cost"] for item in metrics])
            ),
        }
    return {
        "method_aggregate": summaries,
        "joint_gate": {
            "comparisons": dominance_rows,
            "passed_comparisons": sum(
                bool(row["beats_joint_gate"]) for row in dominance_rows
            ),
            "total_comparisons": len(dominance_rows),
            "all_baselines_all_pairs_all_lengths": all(
                bool(row["beats_joint_gate"]) for row in dominance_rows
            ),
            "contract": (
                "Constrained must be endpoint/length/artist valid, no worse on broad "
                "destination progress, and strictly lower on the frozen joint objective "
                "for every baseline, pair, and requested length."
            ),
        },
        "componentwise_dominance_diagnostic": {
            "passed_comparisons": sum(
                bool(row["componentwise_dominates"]) for row in dominance_rows
            ),
            "total_comparisons": len(dominance_rows),
            "all_components_all_comparisons": all(
                bool(row["componentwise_dominates"]) for row in dominance_rows
            ),
            "note": (
                "This stricter diagnostic is not the primary joint gate. Independent "
                "slerp is the per-layer corridor-error optimum, so a different path can "
                "improve continuity only by accepting some corridor-error tradeoff."
            ),
        },
    }


def write_qualitative(path: Path, pairs: Sequence[dict[str, object]]) -> None:
    lines = [
        "# Journey A To B Qualitative Paths",
        "",
        "Embeddings score every route. Artist labels are used only by the declared spacing constraint.",
        "",
    ]
    for pair in pairs:
        lines += [f"## {pair['id']}", ""]
        if pair["graph_shortest"] is None:
            lines += ["### stored_k5_graph_shortest", "", "Unreachable", ""]
        else:
            lines += ["### stored_k5_graph_shortest", ""]
            for position, track in enumerate(pair["graph_shortest"]["tracks"]):
                lines.append(
                    f"{position + 1}. {track['artist'] or '?'} - {track['title'] or '?'}"
                )
            lines.append("")
        for length_record in pair["lengths"]:
            length = length_record["requested_length"]
            for name, record in length_record["methods"].items():
                metrics = record["metrics"]
                lines += [
                    f"### {name} / {length} tracks",
                    "",
                    f"adjacent mean/min {metrics['mean_adjacent_cosine']:.4f} / "
                    f"{metrics['minimum_adjacent_cosine']:.4f}; destination violations "
                    f"{metrics['destination_monotonic_violations']}; corridor mean/max "
                    f"{metrics['mean_interpolation_angle_error_degrees']:.2f} / "
                    f"{metrics['maximum_interpolation_angle_error_degrees']:.2f} deg",
                    "",
                ]
                for position, track in enumerate(record["tracks"]):
                    lines.append(
                        f"{position + 1}. {track['artist'] or '?'} - {track['title'] or '?'}"
                    )
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=active.DEFAULT_DB)
    parser.add_argument("--active-catalog", type=Path, default=active.DEFAULT_ACTIVE_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total_started = time.perf_counter()
    library, adjacency, db_hash, graph_info = load_active_library_and_graph(
        args.db, args.active_catalog, verify_hash=not args.skip_db_hash
    )
    load_seconds = time.perf_counter() - total_started
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    required_ids = {track_id for pair in ROUTE_PAIRS for track_id in pair}
    missing = required_ids - set(by_id)
    if missing:
        raise ValueError(f"route endpoint IDs are not active: {sorted(missing)}")

    evaluation_started = time.perf_counter()
    pairs: list[dict[str, object]] = []
    for pair_position, (start_id, destination_id) in enumerate(ROUTE_PAIRS, start=1):
        start = by_id[start_id]
        destination = by_id[destination_id]
        graph_path = shortest_path(adjacency, start, destination)
        pair_record: dict[str, object] = {
            "id": f"{start_id}_to_{destination_id}",
            "start": track_summary(library, start),
            "destination": track_summary(library, destination),
            "endpoint_cosine": float(
                np.dot(library.embeddings[start], library.embeddings[destination])
            ),
            "graph_shortest": None
            if graph_path is None
            else method_record(
                "stored_k5_graph_shortest",
                graph_path,
                library,
                start,
                destination,
                None,
            ),
            "lengths": [],
        }
        for length in REQUESTED_LENGTHS:
            flexer, flexer_details = flexer_ratio_path(
                library, start, destination, length
            )
            independent = independent_slerp_path(
                library, start, destination, length
            )
            two_part = two_part_path(library, start, destination, length)
            constrained, constrained_details = constrained_beam_path(
                library, start, destination, length
            )
            pair_record["lengths"].append(
                {
                    "requested_length": length,
                    "methods": {
                        "flexer_divergence_ratio": method_record(
                            "flexer_divergence_ratio",
                            flexer,
                            library,
                            start,
                            destination,
                            length,
                            flexer_details,
                        ),
                        "independent_slerp": method_record(
                            "independent_slerp",
                            independent,
                            library,
                            start,
                            destination,
                            length,
                        ),
                        "two_part_nearest": method_record(
                            "two_part_nearest",
                            two_part,
                            library,
                            start,
                            destination,
                            length,
                        ),
                        "constrained_beam": method_record(
                            "constrained_beam",
                            constrained,
                            library,
                            start,
                            destination,
                            length,
                            constrained_details,
                        ),
                    },
                }
            )
            print(
                f"pair {pair_position}/{len(ROUTE_PAIRS)} {start_id}->{destination_id} "
                f"length={length}",
                flush=True,
            )
        pairs.append(pair_record)

    summary = aggregate(pairs)
    evaluation_seconds = time.perf_counter() - evaluation_started
    definitions = {
        "route_pairs": ROUTE_PAIRS,
        "requested_lengths_endpoints_included": REQUESTED_LENGTHS,
        "flexer_far_percent": FLEXER_FAR_PERCENT,
        "candidate_width": CANDIDATE_WIDTH,
        "beam_width": BEAM_WIDTH,
        "branch_width": BRANCH_WIDTH,
        "broad_destination_backstep": MAX_BROAD_DESTINATION_BACKSTEP,
        "outlier_angle_degrees": OUTLIER_ANGLE_DEGREES,
        "weights": {
            "edge": EDGE_WEIGHT,
            "corridor": CORRIDOR_WEIGHT,
            "cross_track": CROSS_TRACK_WEIGHT,
            "backward": BACKWARD_WEIGHT,
            "outlier": OUTLIER_WEIGHT,
            "max_edge": MAX_EDGE_WEIGHT,
            "max_corridor": MAX_CORRIDOR_WEIGHT,
        },
        "artist": {
            "max_per_artist": MAX_PER_ARTIST,
            "min_spacing": MIN_ARTIST_SPACING,
        },
    }
    result: dict[str, object] = {
        "schema": "journey-constrained-eval-v1",
        "scope": {
            "active_tracks": library.count,
            "ranking_and_path_score": "CLaMP3 embeddings only",
            "artist_metadata": "explicit spacing constraint and labels only",
            "playback": "not performed",
        },
        "inputs": {
            "database": str(args.db.resolve()),
            "database_sha256": db_hash,
            "active_catalog": str(args.active_catalog.resolve()),
            "active_catalog_sha256": sha256_file(args.active_catalog),
            "graph": graph_info,
        },
        "source_reference": {
            "paper": "Flexer, Schnitzer, Gasser, Widmer, ISMIR 2008",
            "url": "https://www.cp.jku.at/research/papers/Flexer_etal_ISMIR_2008.pdf",
            "boundary": (
                "The paper motivates the divergence-ratio baseline; this experiment's "
                "angular-distance adaptation is not a reproduction of its MFCC Gaussian model."
            ),
        },
        "definitions": definitions,
        "definitions_sha256": sha256_json(definitions),
        "identity_contract": (
            "No active track ID repeats. The legacy active projection has no decoded-PCM "
            "identity receipts, so embedding or metadata resemblance is not suppressed."
        ),
        "summary": summary,
        "pairs": pairs,
    }
    result["deterministic_payload_sha256"] = sha256_json(result)

    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.json"
    qualitative_path = args.output / "qualitative-paths.md"
    atomic_json(result_path, result)
    write_qualitative(qualitative_path, pairs)
    manifest = {
        "schema": "journey-constrained-run-manifest-v1",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "runtime_seconds": {
            "load_and_active_graph": load_seconds,
            "evaluation": evaluation_seconds,
            "total": time.perf_counter() - total_started,
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "deterministic_payload_sha256": result["deterministic_payload_sha256"],
        "artifacts": {
            "results.json": sha256_file(result_path),
            "qualitative-paths.md": sha256_file(qualitative_path),
        },
    }
    atomic_json(args.output / "run-manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
