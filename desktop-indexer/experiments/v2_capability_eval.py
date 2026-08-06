#!/usr/bin/env python3
"""Evaluate hubness correction, local facets, and deterministic music journeys."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import platform
import sqlite3
import struct
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

import v2_queue_eval as queue_eval


DEFAULT_OUTPUT = (
    queue_eval.REPO_ROOT
    / "desktop-indexer"
    / "audit_raw_data"
    / "v2-discovery"
    / "capabilities"
)
DENSITY_SEEDS = 128
DENSITY_BETAS = (0.0, 0.1, 0.25, 0.5)
FACET_SEED_IDS = (13384, 209, 25130, 21972, 2682, 5831, 31830, 74859)
ROUTE_PAIRS = (
    (13384, 2682),
    (209, 5831),
    (26499, 2955),
    (3201, 28891),
    (33431, 3201),
)


def load_graph(db_path: Path, library: queue_eval.Library) -> tuple[np.ndarray, np.ndarray]:
    with queue_eval.read_only_connection(db_path.resolve()) as connection:
        row = connection.execute(
            "SELECT data FROM binary_data WHERE key = 'knn_graph'"
        ).fetchone()
    if row is None:
        raise ValueError("database has no knn_graph")
    blob = memoryview(row[0])
    n, k = struct.unpack_from("<II", blob, 0)
    if n != library.count:
        raise ValueError(f"graph node count {n} != library {library.count}")
    ids = np.frombuffer(blob, dtype="<i8", count=n, offset=8)
    if not np.array_equal(ids, library.track_ids):
        raise ValueError("graph ID map does not match ordered embedding library")
    entry_dtype = np.dtype([("neighbor", "<u4"), ("weight", "<f4")])
    offset = 8 + n * 8
    entries = np.frombuffer(blob, dtype=entry_dtype, count=n * k, offset=offset).reshape(n, k)
    neighbors = entries["neighbor"].astype(np.int64)
    weights = entries["weight"].astype(np.float32)
    if np.any(neighbors < 0) or np.any(neighbors >= n):
        raise ValueError("invalid graph neighbor")
    return neighbors, weights


def local_density(
    library: queue_eval.Library, neighbors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    result = np.empty(library.count, dtype=np.float32)
    edge_cosines = np.empty(neighbors.shape, dtype=np.float32)
    for start in range(0, library.count, 2_048):
        end = min(start + 2_048, library.count)
        source = library.embeddings[start:end, None, :]
        targets = library.embeddings[neighbors[start:end]]
        edge_cosines[start:end] = np.sum(source * targets, axis=2)
        result[start:end] = np.mean(edge_cosines[start:end], axis=1)
    return result, edge_cosines


def top_by_scores(
    scores: np.ndarray,
    library: queue_eval.Library,
    count: int,
    exclude_indices: set[int] | None = None,
) -> np.ndarray:
    values = scores.copy()
    if exclude_indices:
        values[np.fromiter(exclude_indices, dtype=np.int64)] = -np.inf
    partial = np.argpartition(values, -count)[-count:]
    order = np.lexsort((library.track_ids[partial], -values[partial]))
    return partial[order].astype(np.int64, copy=False)


def gini(values: Sequence[int]) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0 or np.sum(array) == 0:
        return 0.0
    indices = np.arange(1, array.size + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(indices * array) / (array.size * np.sum(array)))
        - (array.size + 1.0) / array.size
    )


def density_experiment(
    library: queue_eval.Library,
    neighbors: np.ndarray,
    density: np.ndarray,
) -> dict[str, object]:
    eligible = np.flatnonzero(
        (library.durations_ms >= 90_000)
        & (library.durations_ms <= 600_000)
        & np.fromiter((bool(artist) for artist in library.artists), dtype=bool)
    )
    seeds = queue_eval.stable_sample(
        library,
        eligible,
        DENSITY_SEEDS,
        salt="v2-capability-density-v1",
    )
    records: list[dict[str, object]] = []
    appearance: dict[float, Counter[int]] = {beta: Counter() for beta in DENSITY_BETAS}
    for position, seed_index in enumerate(seeds, start=1):
        cosine = library.embeddings @ library.embeddings[seed_index]
        raw = top_by_scores(cosine, library, 50, {seed_index})
        exact_top5 = raw[: neighbors.shape[1]]
        stored_top5 = neighbors[seed_index]
        variants: dict[str, object] = {}
        for beta in DENSITY_BETAS:
            score = cosine - beta * density
            chosen = top_by_scores(score, library, 50, {seed_index})
            for index in chosen:
                appearance[beta][int(index)] += 1
            reciprocal = sum(seed_index in neighbors[index] for index in chosen)
            variants[str(beta)] = {
                "track_ids": [int(library.track_ids[index]) for index in chosen],
                "mean_seed_cosine": float(np.mean(cosine[chosen])),
                "min_seed_cosine": float(np.min(cosine[chosen])),
                "mean_candidate_density": float(np.mean(density[chosen])),
                "raw_overlap": len(set(chosen.tolist()) & set(raw.tolist())),
                "stored_k5_reciprocal_count": reciprocal,
            }
        records.append(
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "stored_graph_top5": {
                    "exact_track_ids": [
                        int(library.track_ids[index]) for index in exact_top5
                    ],
                    "stored_track_ids": [
                        int(library.track_ids[index]) for index in stored_top5
                    ],
                    "overlap": len(
                        set(exact_top5.tolist()) & set(stored_top5.tolist())
                    ),
                    "same_positions": int(np.count_nonzero(exact_top5 == stored_top5)),
                },
                "variants": variants,
            }
        )
        if position % 16 == 0:
            print(f"density {position}/{len(seeds)}", flush=True)

    aggregate: dict[str, object] = {}
    for beta in DENSITY_BETAS:
        rows = [record["variants"][str(beta)] for record in records]
        counts = list(appearance[beta].values())
        all_counts = [appearance[beta].get(index, 0) for index in range(library.count)]
        aggregate[str(beta)] = {
            "mean_seed_cosine": float(np.mean([row["mean_seed_cosine"] for row in rows])),
            "mean_min_seed_cosine": float(np.mean([row["min_seed_cosine"] for row in rows])),
            "mean_candidate_density": float(
                np.mean([row["mean_candidate_density"] for row in rows])
            ),
            "mean_raw_overlap": float(np.mean([row["raw_overlap"] for row in rows])),
            "mean_stored_k5_reciprocal_count": float(
                np.mean([row["stored_k5_reciprocal_count"] for row in rows])
            ),
            "unique_results": len(appearance[beta]),
            "result_frequency_gini_nonzero": gini(counts),
            "result_frequency_gini_all_library": gini(all_counts),
            "max_result_frequency": max(counts),
        }
    graph_rows = [record["stored_graph_top5"] for record in records]
    return {
        "seed_count": len(seeds),
        "cohort_track_ids": [int(library.track_ids[index]) for index in seeds],
        "cohort_sha256": hashlib.sha256(
            np.asarray([library.track_ids[index] for index in seeds], dtype="<i8").tobytes()
        ).hexdigest(),
        "formula": "cosine(seed,candidate) - beta * mean_cosine(candidate, stored_k5)",
        "stored_graph_validation": {
            "mean_exact_top5_overlap": float(
                np.mean([row["overlap"] for row in graph_rows])
            ),
            "mean_exact_top5_same_positions": float(
                np.mean([row["same_positions"] for row in graph_rows])
            ),
            "rows_with_exact_top5_set": sum(row["overlap"] == 5 for row in graph_rows),
            "rows_with_exact_top5_order": sum(
                row["same_positions"] == 5 for row in graph_rows
            ),
        },
        "aggregate": aggregate,
        "records": records,
    }


def identity_exclusions(library: queue_eval.Library, seed_index: int) -> set[int]:
    metadata = library.metadata_keys[seed_index]
    path = library.file_paths[seed_index].casefold()
    return {
        index
        for index in range(library.count)
        if index == seed_index
        or library.metadata_keys[index] == metadata
        or library.file_paths[index].casefold() == path
    }


def deterministic_kmedoids(
    embeddings: np.ndarray,
    track_ids: np.ndarray,
    branch_count: int,
    max_iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, int]:
    count = embeddings.shape[0]
    if count < branch_count:
        raise ValueError("not enough local neighbors")
    medoids = [0]
    while len(medoids) < branch_count:
        similarity = embeddings @ embeddings[np.asarray(medoids)].T
        nearest = np.max(similarity, axis=1)
        nearest[np.asarray(medoids)] = np.inf
        minimum = np.min(nearest)
        choices = np.flatnonzero(nearest == minimum)
        chosen = choices[np.argmin(track_ids[choices])]
        medoids.append(int(chosen))

    assignments = np.zeros(count, dtype=np.int64)
    for iteration in range(1, max_iterations + 1):
        similarities = embeddings @ embeddings[np.asarray(medoids)].T
        assignments = np.argmax(similarities, axis=1)
        updated: list[int] = []
        for branch in range(branch_count):
            members = np.flatnonzero(assignments == branch)
            if members.size == 0:
                nearest = np.max(similarities, axis=1)
                choices = np.flatnonzero(nearest == np.min(nearest))
                updated.append(int(choices[np.argmin(track_ids[choices])]))
                continue
            within = embeddings[members] @ embeddings[members].T
            quality = np.mean(within, axis=1)
            best_quality = np.max(quality)
            choices = members[np.flatnonzero(quality == best_quality)]
            updated.append(int(choices[np.argmin(track_ids[choices])]))
        if updated == medoids:
            return np.asarray(medoids), assignments, iteration
        medoids = updated
    similarities = embeddings @ embeddings[np.asarray(medoids)].T
    return np.asarray(medoids), np.argmax(similarities, axis=1), max_iterations


def facet_experiment(library: queue_eval.Library) -> dict[str, object]:
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    records: list[dict[str, object]] = []
    for seed_id in FACET_SEED_IDS:
        seed_index = by_id[seed_id]
        cosine = library.embeddings @ library.embeddings[seed_index]
        local = top_by_scores(cosine, library, 200, identity_exclusions(library, seed_index))
        local_embeddings = library.embeddings[local]
        local_ids = library.track_ids[local]
        medoids, assignments, iterations = deterministic_kmedoids(
            local_embeddings, local_ids, 4
        )
        repeat_medoids, repeat_assignments, _ = deterministic_kmedoids(
            local_embeddings, local_ids, 4
        )
        deterministic = np.array_equal(medoids, repeat_medoids) and np.array_equal(
            assignments, repeat_assignments
        )
        branches: list[dict[str, object]] = []
        for branch in range(4):
            members_local = np.flatnonzero(assignments == branch)
            members = local[members_local]
            medoid_index = int(local[medoids[branch]])
            medoid_similarity = library.embeddings[members] @ library.embeddings[medoid_index]
            order = np.lexsort((library.track_ids[members], -medoid_similarity))
            ordered_members = members[order]
            pairwise = library.embeddings[members] @ library.embeddings[members].T
            branches.append(
                {
                    "medoid": queue_eval.track_summary(library, medoid_index),
                    "size": int(members.size),
                    "mean_seed_cosine": float(np.mean(cosine[members])),
                    "min_seed_cosine": float(np.min(cosine[members])),
                    "mean_within_cosine": float(np.mean(pairwise)),
                    "top_tracks": [
                        queue_eval.track_summary(library, int(index))
                        for index in ordered_members[:12]
                    ],
                }
            )
        records.append(
            {
                "seed": queue_eval.track_summary(library, seed_index),
                "local_count": int(local.size),
                "iterations": iterations,
                "repeat_deterministic": deterministic,
                "medoid_track_ids": [
                    int(library.track_ids[local[index]]) for index in medoids
                ],
                "branches": branches,
            }
        )
        print(f"facets {library.artists[seed_index]} - {library.titles[seed_index]}", flush=True)
    return {"neighbor_count": 200, "branch_count": 4, "records": records}


def undirected_graph(
    neighbors: np.ndarray, edge_cosines: np.ndarray
) -> list[dict[int, float]]:
    adjacency: list[dict[int, float]] = [dict() for _ in range(neighbors.shape[0])]
    for source in range(neighbors.shape[0]):
        for edge, target in enumerate(neighbors[source]):
            target_int = int(target)
            cosine = float(edge_cosines[source, edge])
            cost = math.acos(max(-1.0, min(1.0, cosine)))
            previous = adjacency[source].get(target_int)
            if previous is None or cost < previous:
                adjacency[source][target_int] = cost
                adjacency[target_int][source] = cost
    return adjacency


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
            # Heap entries already use the node index as a stable tie-breaker.
            # Re-parenting equal-cost paths can form predecessor cycles across
            # zero-cost duplicate edges, so only strict improvements are valid.
            if old is None or candidate < old - 1e-15:
                distance[neighbor] = candidate
                previous[neighbor] = node
                heapq.heappush(queue, (candidate, neighbor))
    return None


def slerp(start: np.ndarray, end: np.ndarray, fraction: float) -> np.ndarray:
    cosine = float(np.clip(start @ end, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-6:
        result = (1.0 - fraction) * start + fraction * end
    else:
        sine = math.sin(angle)
        result = (
            math.sin((1.0 - fraction) * angle) / sine * start
            + math.sin(fraction * angle) / sine * end
        )
    return result / np.linalg.norm(result)


def journey_path(
    library: queue_eval.Library,
    start: int,
    destination: int,
    queue_count: int = 12,
) -> tuple[list[int], list[float]]:
    selected: list[int] = []
    interpolation_scores: list[float] = []
    used = identity_exclusions(library, start) | identity_exclusions(library, destination)
    destination_artist = queue_eval.normalized_artist(library.artists[destination])
    for position in range(1, queue_count):
        query = slerp(
            library.embeddings[start],
            library.embeddings[destination],
            position / queue_count,
        )
        scores = library.embeddings @ query
        candidates = top_by_scores(scores, library, 512, used | set(selected))
        chosen: int | None = None
        for candidate in candidates:
            candidate_int = int(candidate)
            if position >= queue_count - 3 and destination_artist is not None:
                if queue_eval.normalized_artist(library.artists[candidate_int]) == destination_artist:
                    continue
            if queue_eval.can_add_artist(
                candidate_int,
                selected,
                library,
                queue_eval.DEFAULT_MAX_PER_ARTIST,
                queue_eval.DEFAULT_MIN_ARTIST_SPACING,
            ):
                chosen = candidate_int
                break
        if chosen is None:
            raise RuntimeError("journey candidate pool exhausted")
        selected.append(chosen)
        interpolation_scores.append(float(scores[chosen]))
    selected.append(destination)
    interpolation_scores.append(1.0)
    return selected, interpolation_scores


def path_metrics(
    library: queue_eval.Library,
    start: int,
    destination: int,
    path_without_start: Sequence[int],
    interpolation_scores: Sequence[float] | None = None,
) -> dict[str, object]:
    full = [start] + list(path_without_start)
    embeddings = library.embeddings[np.asarray(full)]
    adjacent = np.sum(embeddings[:-1] * embeddings[1:], axis=1)
    destination_progress = embeddings @ library.embeddings[destination]
    start_progress = embeddings @ library.embeddings[start]
    artists = [queue_eval.normalized_artist(library.artists[index]) for index in path_without_start]
    valid = all(
        queue_eval.can_add_artist(
            index,
            list(path_without_start[:position]),
            library,
            queue_eval.DEFAULT_MAX_PER_ARTIST,
            queue_eval.DEFAULT_MIN_ARTIST_SPACING,
        )
        for position, index in enumerate(path_without_start)
    )
    return {
        "positions_excluding_start": len(path_without_start),
        "mean_adjacent_cosine": float(np.mean(adjacent)),
        "min_adjacent_cosine": float(np.min(adjacent)),
        "destination_progress": destination_progress.tolist(),
        "destination_monotonic_violations": int(
            np.count_nonzero(np.diff(destination_progress) < -1e-6)
        ),
        "start_progress": start_progress.tolist(),
        "start_monotonic_violations": int(np.count_nonzero(np.diff(start_progress) > 1e-6)),
        "artist_constraints_valid": valid,
        "known_artist_count": sum(artist is not None for artist in artists),
        "interpolation_scores": list(interpolation_scores) if interpolation_scores else None,
    }


def route_experiment(
    library: queue_eval.Library, neighbors: np.ndarray, edge_cosines: np.ndarray
) -> dict[str, object]:
    by_id = {int(track_id): index for index, track_id in enumerate(library.track_ids)}
    adjacency = undirected_graph(neighbors, edge_cosines)
    records: list[dict[str, object]] = []
    for start_id, destination_id in ROUTE_PAIRS:
        start = by_id[start_id]
        destination = by_id[destination_id]
        graph = shortest_path(adjacency, start, destination)
        journey, interpolation_scores = journey_path(library, start, destination)
        records.append(
            {
                "start": queue_eval.track_summary(library, start),
                "destination": queue_eval.track_summary(library, destination),
                "graph_shortest": None
                if graph is None
                else {
                    "metrics": path_metrics(library, start, destination, graph[1:]),
                    "tracks": [queue_eval.track_summary(library, index) for index in graph],
                },
                "interpolated_journey": {
                    "metrics": path_metrics(
                        library,
                        start,
                        destination,
                        journey,
                        interpolation_scores,
                    ),
                    "tracks": [queue_eval.track_summary(library, start)]
                    + [queue_eval.track_summary(library, index) for index in journey],
                },
            }
        )
        print(f"route {start_id}->{destination_id}", flush=True)
    return {
        "pairs": records,
        "undirected_component_note": "graph shortest is null when endpoints are disconnected",
    }


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_qualitative(path: Path, result: dict[str, object]) -> None:
    lines = ["# Capability Qualitative Packet", "", "## Local Facets", ""]
    for record in result["facets"]["records"]:
        seed = record["seed"]
        lines.extend((f"### {seed['artist']} - {seed['title']}", ""))
        for number, branch in enumerate(record["branches"], start=1):
            medoid = branch["medoid"]
            lines.append(
                f"Branch {number} ({branch['size']}): {medoid['artist']} - {medoid['title']}"
            )
            for track in branch["top_tracks"][:8]:
                lines.append(f"- {track['artist'] or '?'} - {track['title'] or '?'}")
            lines.append("")
    lines.extend(("## Routes", ""))
    for record in result["routes"]["pairs"]:
        start = record["start"]
        destination = record["destination"]
        lines.append(
            f"### {start['artist']} - {start['title']} -> "
            f"{destination['artist']} - {destination['title']}"
        )
        lines.append("")
        for name in ("graph_shortest", "interpolated_journey"):
            route = record[name]
            lines.append(f"{name}:")
            if route is None:
                lines.append("- unreachable")
            else:
                for track in route["tracks"]:
                    lines.append(f"- {track['artist'] or '?'} - {track['title'] or '?'}")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=queue_eval.DEFAULT_DB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-db-hash", action="store_true", help="development only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    library, db_hash = queue_eval.load_library(
        args.db, verify_hash=not args.skip_db_hash
    )
    neighbors, weights = load_graph(args.db, library)
    density, edge_cosines = local_density(library, neighbors)
    result = {
        "database": {
            "path": str(args.db.resolve()),
            "sha256": db_hash,
            "tracks": library.count,
            "dim": library.dim,
        },
        "graph": {
            "nodes": int(neighbors.shape[0]),
            "k": int(neighbors.shape[1]),
            "mean_weight_l1_from_uniform": float(
                np.mean(np.sum(np.abs(weights - 1.0 / weights.shape[1]), axis=1))
            ),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "density": density_experiment(library, neighbors, density),
        "facets": facet_experiment(library),
        "routes": route_experiment(library, neighbors, edge_cosines),
    }
    result["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(args.output / "results.json", result)
    write_qualitative(args.output / "qualitative.md", result)
    print(f"Complete. Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
