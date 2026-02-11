"""Algorithm audit: exhaustive validation of all recommendation algorithms.

Faithfully ports every algorithm from the Kotlin Android code to Python/numpy,
then runs them against the real embedding database to validate correctness,
monotonicity, diversity properties, and edge-case behavior.

Usage:
    poweramp-indexer audit embeddings.db [--seeds N] [--quick]
"""

from __future__ import annotations

import json
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .database import EmbeddingDatabase


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class AuditCorpus:
    """All data needed for the audit loaded into memory."""
    embeddings: np.ndarray          # (N, D) float32, L2-normalized
    track_ids: np.ndarray           # (N,) int64
    artists: dict[int, str | None]  # track_id -> artist (lowercase or None)
    titles: dict[int, str | None]   # track_id -> title
    tid_to_idx: dict[int, int]      # track_id -> row index
    idx_to_tid: dict[int, int]      # row index -> track_id
    graph_n: int = 0
    graph_k: int = 0
    graph_ids: np.ndarray | None = None      # (graph_n,) int64
    graph_neighbors: np.ndarray | None = None  # (graph_n, graph_k) int32 indices into graph_ids
    graph_weights: np.ndarray | None = None    # (graph_n, graph_k) float32
    graph_tid_to_idx: dict[int, int] = field(default_factory=dict)
    cluster_centroids: np.ndarray | None = None  # (K, D) float32
    cluster_labels: np.ndarray | None = None     # (N,) int32


def load_corpus(db: EmbeddingDatabase, on_progress=None) -> AuditCorpus:
    """Load all fused embeddings, metadata, and kNN graph from the database."""

    def progress(msg):
        if on_progress:
            on_progress(msg)

    # Load fused embeddings
    progress("Loading fused embeddings...")
    rows = db.conn.execute(
        "SELECT e.track_id, e.embedding FROM embeddings_fused e"
    ).fetchall()

    if not rows:
        raise ValueError("No fused embeddings found. Run 'fuse' first.")

    n = len(rows)
    dim = len(rows[0]["embedding"]) // 4
    embeddings = np.empty((n, dim), dtype=np.float32)
    track_ids = np.empty(n, dtype=np.int64)

    for i, row in enumerate(rows):
        track_ids[i] = row["track_id"]
        embeddings[i] = np.frombuffer(row["embedding"], dtype=np.float32)

    tid_to_idx = {int(track_ids[i]): i for i in range(n)}
    idx_to_tid = {i: int(track_ids[i]) for i in range(n)}

    progress(f"Loaded {n} embeddings ({dim}d)")

    # Load track metadata
    progress("Loading track metadata...")
    artists = {}
    titles = {}
    meta_rows = db.conn.execute(
        "SELECT id, artist, title FROM tracks"
    ).fetchall()
    for row in meta_rows:
        tid = row["id"]
        artists[tid] = row["artist"].lower() if row["artist"] else None
        titles[tid] = row["title"]

    # Load clusters
    progress("Loading clusters...")
    cluster_centroids = None
    cluster_labels = None

    centroid_rows = db.conn.execute(
        "SELECT cluster_id, embedding FROM clusters ORDER BY cluster_id"
    ).fetchall()
    if centroid_rows:
        k = len(centroid_rows)
        cdim = len(centroid_rows[0]["embedding"]) // 4
        cluster_centroids = np.empty((k, cdim), dtype=np.float32)
        for row in centroid_rows:
            cluster_centroids[row["cluster_id"]] = np.frombuffer(
                row["embedding"], dtype=np.float32
            )

        # Load cluster assignments
        label_rows = db.conn.execute(
            "SELECT id, cluster_id FROM tracks WHERE cluster_id IS NOT NULL"
        ).fetchall()
        cluster_labels = np.full(n, -1, dtype=np.int32)
        for row in label_rows:
            idx = tid_to_idx.get(row["id"])
            if idx is not None and row["cluster_id"] is not None:
                cluster_labels[idx] = row["cluster_id"]

        progress(f"Loaded {k} cluster centroids")

    # Load kNN graph
    progress("Loading kNN graph...")
    graph_blob = db.get_binary("knn_graph")

    corpus = AuditCorpus(
        embeddings=embeddings, track_ids=track_ids, artists=artists,
        titles=titles, tid_to_idx=tid_to_idx, idx_to_tid=idx_to_tid,
        cluster_centroids=cluster_centroids, cluster_labels=cluster_labels,
    )

    if graph_blob:
        offset = 0
        graph_n, graph_k = struct.unpack_from('<II', graph_blob, offset)
        offset += 8

        graph_ids = np.empty(graph_n, dtype=np.int64)
        for i in range(graph_n):
            graph_ids[i] = struct.unpack_from('<q', graph_blob, offset)[0]
            offset += 8

        graph_neighbors = np.empty((graph_n, graph_k), dtype=np.int32)
        graph_weights = np.empty((graph_n, graph_k), dtype=np.float32)
        for i in range(graph_n):
            for j in range(graph_k):
                ni, wi = struct.unpack_from('<If', graph_blob, offset)
                graph_neighbors[i, j] = ni
                graph_weights[i, j] = wi
                offset += 8

        corpus.graph_n = graph_n
        corpus.graph_k = graph_k
        corpus.graph_ids = graph_ids
        corpus.graph_neighbors = graph_neighbors
        corpus.graph_weights = graph_weights
        corpus.graph_tid_to_idx = {int(graph_ids[i]): i for i in range(graph_n)}

        progress(f"Loaded kNN graph: {graph_n} nodes, K={graph_k}")
    else:
        progress("No kNN graph found in database")

    return corpus


# ---------------------------------------------------------------------------
# Algorithm ports (faithful to Kotlin)
# ---------------------------------------------------------------------------

def find_top_k(corpus: AuditCorpus, query: np.ndarray, k: int,
               exclude_ids: set[int] | None = None) -> list[tuple[int, float]]:
    """Brute-force dot product retrieval. Returns list of (track_id, similarity)."""
    sims = corpus.embeddings @ query  # (N,)
    if exclude_ids:
        for tid in exclude_ids:
            idx = corpus.tid_to_idx.get(tid)
            if idx is not None:
                sims[idx] = -np.inf

    # Top-k via argpartition
    if k >= len(sims):
        top_indices = np.argsort(sims)[::-1]
    else:
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

    result = []
    for idx in top_indices:
        if sims[idx] == -np.inf:
            break
        tid = corpus.idx_to_tid[int(idx)]
        result.append((tid, float(sims[idx])))

    return result[:k]


def mmr_select_one(
    candidates: list[tuple[int, float]],
    selected_embs: list[np.ndarray],
    corpus: AuditCorpus,
    lambda_: float
) -> tuple[int, float] | None:
    """MMR single selection — faithful port from MmrSelector.selectOne()."""
    if not candidates:
        return None
    if not selected_embs:
        return candidates[0]

    best_id = -1
    best_relevance = 0.0
    best_mmr = -np.inf

    for (tid, relevance) in candidates:
        idx = corpus.tid_to_idx.get(tid)
        if idx is None:
            continue
        emb = corpus.embeddings[idx]

        max_sim = -np.inf
        for sel in selected_embs:
            sim = float(np.dot(emb, sel))
            if sim > max_sim:
                max_sim = sim

        mmr_score = lambda_ * relevance - (1.0 - lambda_) * max_sim

        if mmr_score > best_mmr:
            best_mmr = mmr_score
            best_relevance = relevance
            best_id = tid

    return (best_id, best_relevance) if best_id >= 0 else None


def mmr_select_batch(
    candidates: list[tuple[int, float]],
    num_select: int,
    corpus: AuditCorpus,
    lambda_: float
) -> list[tuple[int, float]]:
    """Iterative MMR batch selection — faithful port from MmrSelector.selectBatch()."""
    if not candidates:
        return []

    result = []
    selected_embs = []
    remaining = list(candidates)

    # Pre-load embeddings
    emb_cache = {}
    for (tid, _) in candidates:
        idx = corpus.tid_to_idx.get(tid)
        if idx is not None:
            emb_cache[tid] = corpus.embeddings[idx]

    # Track max-sim-to-selected
    candidate_list = list(candidates)
    max_sim_to_selected = np.full(len(candidates), -np.inf)

    # Build tid -> original index map (the HashMap performance fix)
    tid_to_orig = {tid: i for i, (tid, _) in enumerate(candidate_list)}

    for step in range(num_select):
        if not remaining:
            break

        best_idx = -1
        best_score = -np.inf

        for i, (tid, relevance) in enumerate(remaining):
            emb = emb_cache.get(tid)
            if emb is None:
                continue

            orig_idx = tid_to_orig[tid]

            if selected_embs:
                last_selected = selected_embs[-1]
                sim = float(np.dot(emb, last_selected))
                if sim > max_sim_to_selected[orig_idx]:
                    max_sim_to_selected[orig_idx] = sim

            penalty = 0.0 if not selected_embs else max_sim_to_selected[orig_idx]
            mmr_score = lambda_ * relevance - (1.0 - lambda_) * penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx < 0:
            break

        sel_tid, sel_sim = remaining.pop(best_idx)
        sel_emb = emb_cache.get(sel_tid)
        if sel_emb is None:
            continue
        result.append((sel_tid, sel_sim))
        selected_embs.append(sel_emb)

    return result


def dpp_select_batch(
    candidates: list[tuple[int, float]],
    num_select: int,
    corpus: AuditCorpus,
    quality_exponent: float = 1.0
) -> list[tuple[int, float]]:
    """DPP greedy MAP with incremental Cholesky — faithful port from DppSelector.selectBatch()."""
    if not candidates:
        return []

    n = len(candidates)
    k = min(num_select, n)

    # Pre-load embeddings and quality
    embeddings = []
    quality = np.zeros(n, dtype=np.float64)
    valid_mask = np.zeros(n, dtype=bool)

    for i, (tid, relevance) in enumerate(candidates):
        idx = corpus.tid_to_idx.get(tid)
        if idx is not None:
            embeddings.append(corpus.embeddings[idx].astype(np.float64))
            quality[i] = relevance if quality_exponent == 1.0 else relevance ** quality_exponent
            valid_mask[i] = True
        else:
            embeddings.append(np.zeros(0, dtype=np.float64))

    # Greedy DPP MAP
    selected = []
    dim = next((e.shape[0] for e in embeddings if e.shape[0] > 0), 0)
    if dim == 0:
        return []

    cholesky_factors = np.zeros((n, k), dtype=np.float64)
    diag_remaining = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if valid_mask[i]:
            diag_remaining[i] = quality[i] * quality[i]

    selected_set = set()

    for step in range(k):
        best_idx = -1
        best_gain = -1.0

        for i in range(n):
            if not valid_mask[i] or i in selected_set:
                continue
            if diag_remaining[i] > best_gain:
                best_gain = diag_remaining[i]
                best_idx = i

        if best_idx < 0 or best_gain <= 1e-10:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)

        sqrt_gain = np.sqrt(best_gain)

        for i in range(n):
            if not valid_mask[i] or i in selected_set:
                continue

            kernel_val = quality[i] * quality[best_idx] * float(
                np.dot(embeddings[i], embeddings[best_idx])
            )

            subtracted = kernel_val
            for j in range(step):
                subtracted -= cholesky_factors[i, j] * cholesky_factors[best_idx, j]

            cholesky_factors[i, step] = subtracted / sqrt_gain
            diag_remaining[i] -= cholesky_factors[i, step] ** 2
            if diag_remaining[i] < 0:
                diag_remaining[i] = 0

        cholesky_factors[best_idx, step] = sqrt_gain

    return [candidates[idx] for idx in selected]


def _compute_rank_scores(candidates: list[tuple[int, float]]) -> list[float]:
    """Compute rank-based scores: (1 - rank/N) mapped to [0, 1].
    Rank 0 (best) maps to ~1.0, rank N-1 (worst) maps to ~0.0."""
    n = len(candidates)
    if n == 0:
        return []
    # Argsort by score descending
    sorted_indices = sorted(range(n), key=lambda i: candidates[i][1], reverse=True)
    rank_scores = [0.0] * n
    for rank, orig_idx in enumerate(sorted_indices):
        rank_scores[orig_idx] = 1.0 - rank / n
    return rank_scores


def temperature_select_one(
    candidates: list[tuple[int, float]],
    temperature: float,
    rng: np.random.Generator | None = None
) -> tuple[int, float] | None:
    """Gumbel-max single selection with rank-based transform — faithful port from TemperatureSelector.selectOne().

    Uses rank-based scores instead of raw cosine similarity to give the
    temperature knob meaningful dynamic range (raw scores cluster 0.93-0.96).
    """
    if not candidates:
        return None
    if temperature <= 1e-6:
        return max(candidates, key=lambda x: x[1])

    if rng is None:
        rng = np.random.default_rng()

    rank_scores = _compute_rank_scores(candidates)

    best_id = -1
    best_orig_score = 0.0
    best_perturbed = -np.inf

    for i, (tid, score) in enumerate(candidates):
        u = np.clip(rng.random(), 1e-10, 1 - 1e-10)
        gumbel = -np.log(-np.log(u))
        perturbed = rank_scores[i] / temperature + gumbel
        if perturbed > best_perturbed:
            best_perturbed = perturbed
            best_id = tid
            best_orig_score = score

    return (best_id, best_orig_score) if best_id >= 0 else None


def temperature_select_batch(
    candidates: list[tuple[int, float]],
    num_select: int,
    temperature: float,
    rng: np.random.Generator | None = None
) -> list[tuple[int, float]]:
    """Gumbel-max batch (w/o replacement) with rank-based transform — port from TemperatureSelector.selectBatch().

    Recomputes rank scores among remaining candidates each step.
    """
    if not candidates:
        return []
    if temperature <= 1e-6:
        return sorted(candidates, key=lambda x: x[1], reverse=True)[:num_select]

    if rng is None:
        rng = np.random.default_rng()

    remaining = list(candidates)
    result = []

    for step in range(min(num_select, len(candidates))):
        # Recompute rank scores among remaining
        rank_scores = _compute_rank_scores(remaining)

        best_idx = -1
        best_perturbed = -np.inf

        for i, (tid, score) in enumerate(remaining):
            u = np.clip(rng.random(), 1e-10, 1 - 1e-10)
            gumbel = -np.log(-np.log(u))
            perturbed = rank_scores[i] / temperature + gumbel
            if perturbed > best_perturbed:
                best_perturbed = perturbed
                best_idx = i

        if best_idx < 0:
            break
        result.append(remaining.pop(best_idx))

    return result


def personalized_pagerank(
    corpus: AuditCorpus,
    seed_track_id: int,
    alpha: float = 0.5,
    iterations: int = 30
) -> list[tuple[int, float]]:
    """Personalized PageRank — faithful port from RandomWalkSelector.computeRanking()."""
    if corpus.graph_ids is None:
        return []

    graph_n = corpus.graph_n
    graph_k = corpus.graph_k

    seed_graph_idx = corpus.graph_tid_to_idx.get(seed_track_id)
    if seed_graph_idx is None:
        return []

    seed_tid = int(corpus.graph_ids[seed_graph_idx])

    # Sparse PageRank in track ID space
    pi = {seed_tid: 1.0}
    restart = {seed_tid: 1.0}

    for _ in range(iterations):
        new_pi = {}

        for node_tid, prob in pi.items():
            node_gidx = corpus.graph_tid_to_idx.get(node_tid)
            if node_gidx is None:
                continue
            for j in range(graph_k):
                neighbor_gidx = int(corpus.graph_neighbors[node_gidx, j])
                weight = float(corpus.graph_weights[node_gidx, j])
                neighbor_tid = int(corpus.graph_ids[neighbor_gidx])
                new_pi[neighbor_tid] = new_pi.get(neighbor_tid, 0.0) + (1.0 - alpha) * prob * weight

        for seed_id, weight in restart.items():
            new_pi[seed_id] = new_pi.get(seed_id, 0.0) + alpha * weight

        pi = new_pi

        if len(pi) > graph_n // 2:
            pi = {k: v for k, v in pi.items() if v > 1e-8}

    seed_set = {seed_tid}
    results = [(tid, score) for tid, score in pi.items() if tid not in seed_set]
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compute_alpha(base_alpha: float, step: int, total_steps: int, decay: str) -> float:
    """Compute anchor alpha at a given step — port from DriftEngine.computeAlpha()."""
    if total_steps <= 1:
        return base_alpha
    progress = step / (total_steps - 1)

    if decay == "none":
        return base_alpha
    elif decay == "linear":
        return base_alpha * (1.0 - progress)
    elif decay == "exponential":
        return base_alpha * np.exp(-3.0 * progress)
    elif decay == "step":
        return base_alpha if progress < 0.5 else base_alpha * 0.2
    return base_alpha


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def drift_update_query(
    seed_emb: np.ndarray, current_emb: np.ndarray, ema_state: np.ndarray | None,
    step: int, total_steps: int, drift_mode: str, anchor_strength: float,
    anchor_decay: str, momentum_beta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Port of DriftEngine.updateQuery()."""
    if drift_mode == "seed_interpolation":
        alpha = compute_alpha(anchor_strength, step, total_steps, anchor_decay)
        query = l2_normalize(alpha * seed_emb + (1.0 - alpha) * current_emb)
        return query, current_emb
    else:  # momentum
        prev = ema_state if ema_state is not None else seed_emb
        ema = l2_normalize(momentum_beta * prev + (1.0 - momentum_beta) * current_emb)
        return ema, ema


def post_filter(
    tracks: list[tuple[int, float]],
    corpus: AuditCorpus,
    max_per_artist: int,
    min_spacing: int
) -> list[tuple[int, float]]:
    """Post-filter — port from PostFilter.enforceBatch()."""
    result = []
    artist_counts: dict[str, int] = defaultdict(int)

    for (tid, score) in tracks:
        artist = corpus.artists.get(tid)

        if artist is not None:
            if artist_counts[artist] >= max_per_artist:
                continue

        if artist is not None and min_spacing > 0 and result:
            recent = result[-min_spacing:]
            if any(corpus.artists.get(r[0]) == artist for r in recent):
                continue

        result.append((tid, score))
        if artist is not None:
            artist_counts[artist] += 1

    return result


# ---------------------------------------------------------------------------
# Pipelines (batch & drift)
# ---------------------------------------------------------------------------

def run_batch_pipeline(
    seed_tid: int, corpus: AuditCorpus, mode: str,
    pool_size: int = 200, num_tracks: int = 50,
    lambda_: float = 0.6, temperature: float = 0.1,
    alpha: float = 0.5, max_per_artist: int = 3, min_spacing: int = 3,
    rng: np.random.Generator | None = None
) -> list[tuple[int, float]]:
    """Run a full batch pipeline: retrieve -> select -> post-filter."""
    seed_idx = corpus.tid_to_idx.get(seed_tid)
    if seed_idx is None:
        return []
    seed_emb = corpus.embeddings[seed_idx]

    if mode == "random_walk":
        ranking = personalized_pagerank(corpus, seed_tid, alpha=alpha)
        tracks = ranking[:pool_size]
        return post_filter(tracks, corpus, max_per_artist, min_spacing)[:num_tracks]

    candidates = find_top_k(corpus, seed_emb, pool_size, exclude_ids={seed_tid})

    if mode == "mmr":
        selected = mmr_select_batch(candidates, num_tracks, corpus, lambda_)
    elif mode == "dpp":
        selected = dpp_select_batch(candidates, num_tracks, corpus)
    elif mode == "temperature":
        selected = temperature_select_batch(candidates, num_tracks, temperature, rng=rng)
    else:
        selected = candidates[:num_tracks]

    return post_filter(selected, corpus, max_per_artist, min_spacing)


def run_drift_pipeline(
    seed_tid: int, corpus: AuditCorpus, mode: str,
    drift_mode: str = "seed_interpolation", pool_size: int = 50,
    num_tracks: int = 50, lambda_: float = 0.6, temperature: float = 0.1,
    anchor_strength: float = 0.5, anchor_decay: str = "exponential",
    momentum_beta: float = 0.7, max_per_artist: int = 3, min_spacing: int = 3,
    rng: np.random.Generator | None = None
) -> list[tuple[int, float]]:
    """Run a drift pipeline: per-step retrieve -> select -> query update -> post-filter."""
    seed_idx = corpus.tid_to_idx.get(seed_tid)
    if seed_idx is None:
        return []
    seed_emb = corpus.embeddings[seed_idx].astype(np.float64)
    query = seed_emb.copy()
    ema_state = None

    result = []
    selected_tracks = []  # (tid, artist) for post-filter
    selected_embs = []
    seen = {seed_tid}

    for step in range(num_tracks):
        candidates = find_top_k(corpus, query.astype(np.float32), pool_size, exclude_ids=seen)
        if not candidates:
            break

        # Select one
        if mode == "mmr":
            pick = mmr_select_one(candidates, selected_embs, corpus, lambda_)
        elif mode == "dpp":
            picks = dpp_select_batch(candidates, 1, corpus)
            pick = picks[0] if picks else None
        elif mode == "temperature":
            pick = temperature_select_one(candidates, temperature, rng=rng)
        else:
            pick = candidates[0] if candidates else None

        if pick is None:
            break

        tid, score = pick
        seen.add(tid)

        # Post-filter check
        artist = corpus.artists.get(tid)
        can_add = True
        if artist is not None:
            count = sum(1 for t in selected_tracks if corpus.artists.get(t) == artist)
            if count >= max_per_artist:
                can_add = False
            if min_spacing > 0 and selected_tracks:
                recent = selected_tracks[-min_spacing:]
                if any(corpus.artists.get(t) == artist for t in recent):
                    can_add = False

        if not can_add:
            # Fallback
            for alt_tid, alt_score in candidates:
                if alt_tid == tid or alt_tid in seen:
                    continue
                alt_artist = corpus.artists.get(alt_tid)
                alt_ok = True
                if alt_artist is not None:
                    alt_count = sum(1 for t in selected_tracks if corpus.artists.get(t) == alt_artist)
                    if alt_count >= max_per_artist:
                        alt_ok = False
                    if min_spacing > 0 and selected_tracks:
                        recent = selected_tracks[-min_spacing:]
                        if any(corpus.artists.get(t) == alt_artist for t in recent):
                            alt_ok = False
                if alt_ok:
                    tid, score = alt_tid, alt_score
                    seen.add(tid)
                    can_add = True
                    break

        if not can_add:
            continue

        result.append((tid, score))
        selected_tracks.append(tid)
        pick_idx = corpus.tid_to_idx.get(tid)
        if pick_idx is not None:
            pick_emb = corpus.embeddings[pick_idx].astype(np.float64)
            selected_embs.append(corpus.embeddings[pick_idx])

            new_query, new_ema = drift_update_query(
                seed_emb, pick_emb, ema_state, step, num_tracks,
                drift_mode, anchor_strength, anchor_decay, momentum_beta
            )
            query = new_query
            ema_state = new_ema

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class RunMetrics:
    """Metrics for a single audit run."""
    seed_tid: int = 0
    mode: str = ""
    config: dict = field(default_factory=dict)
    queue_length: int = 0
    mean_sim_to_seed: float = 0.0
    sim_at_position: list[float] = field(default_factory=list)
    mean_pairwise_sim: float = 0.0
    consecutive_sim: float = 0.0
    unique_artists: int = 0
    artist_hhi: float = 0.0
    transitive_count: int = 0  # Random Walk only
    selected_tids: list[int] = field(default_factory=list)


def compute_metrics(
    seed_tid: int, results: list[tuple[int, float]], corpus: AuditCorpus,
    mode: str = "", config: dict | None = None
) -> RunMetrics:
    """Compute all metrics for a single run."""
    m = RunMetrics(seed_tid=seed_tid, mode=mode, config=config or {})
    m.selected_tids = [tid for tid, _ in results]
    m.queue_length = len(results)

    if not results:
        return m

    seed_idx = corpus.tid_to_idx.get(seed_tid)
    if seed_idx is None:
        return m
    seed_emb = corpus.embeddings[seed_idx]

    # Similarity to seed at each position
    result_embs = []
    for tid, _ in results:
        idx = corpus.tid_to_idx.get(tid)
        if idx is not None:
            emb = corpus.embeddings[idx]
            result_embs.append(emb)
            m.sim_at_position.append(float(np.dot(emb, seed_emb)))
        else:
            result_embs.append(None)
            m.sim_at_position.append(0.0)

    m.mean_sim_to_seed = float(np.mean(m.sim_at_position))

    # Mean pairwise similarity
    valid_embs = [e for e in result_embs if e is not None]
    if len(valid_embs) >= 2:
        E = np.stack(valid_embs)
        pw = E @ E.T
        n_valid = len(valid_embs)
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n_valid, k=1)
        m.mean_pairwise_sim = float(np.mean(pw[triu_indices]))

    # Consecutive similarity
    if len(valid_embs) >= 2:
        consec_sims = []
        prev = valid_embs[0]
        for e in valid_embs[1:]:
            consec_sims.append(float(np.dot(prev, e)))
            prev = e
        m.consecutive_sim = float(np.mean(consec_sims))

    # Artist diversity
    artists_in_result = [corpus.artists.get(tid) for tid, _ in results]
    known_artists = [a for a in artists_in_result if a is not None]
    m.unique_artists = len(set(known_artists))

    # Herfindahl-Hirschman Index
    if known_artists:
        counts = defaultdict(int)
        for a in known_artists:
            counts[a] += 1
        total = len(known_artists)
        m.artist_hhi = sum((c / total) ** 2 for c in counts.values())

    return m


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def select_seeds(corpus: AuditCorpus, n_seeds: int = 200) -> list[int]:
    """Select diverse seed tracks using cluster centroids."""
    seeds = []

    if corpus.cluster_centroids is not None and corpus.cluster_labels is not None:
        n_clusters = len(corpus.cluster_centroids)
        for k in range(min(n_clusters, n_seeds)):
            mask = corpus.cluster_labels == k
            if not mask.any():
                continue
            cluster_embs = corpus.embeddings[mask]
            centroid = corpus.cluster_centroids[k]
            sims = cluster_embs @ centroid
            best_local = np.argmax(sims)
            global_idx = np.where(mask)[0][best_local]
            seeds.append(int(corpus.track_ids[global_idx]))

    # Fill remaining with random if needed
    if len(seeds) < n_seeds:
        rng = np.random.default_rng(42)
        all_tids = set(corpus.track_ids.tolist())
        existing = set(seeds)
        remaining = list(all_tids - existing)
        rng.shuffle(remaining)
        seeds.extend(remaining[:n_seeds - len(seeds)])

    return seeds[:n_seeds]


# ---------------------------------------------------------------------------
# Validation assertions
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    name: str
    passed: bool
    detail: str = ""


def validate_post_filter(
    results: list[tuple[int, float]], corpus: AuditCorpus,
    max_per_artist: int, min_spacing: int
) -> bool:
    """Check post-filter constraints hold for a result list."""
    artist_counts: dict[str, int] = defaultdict(int)

    for i, (tid, _) in enumerate(results):
        artist = corpus.artists.get(tid)
        if artist is None:
            continue

        if artist_counts[artist] >= max_per_artist:
            return False

        if min_spacing > 0 and i > 0:
            start = max(0, i - min_spacing)
            recent_artists = [corpus.artists.get(results[j][0]) for j in range(start, i)]
            if artist in recent_artists:
                return False

        artist_counts[artist] += 1

    return True


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def run_audit(db_path: Path, n_seeds: int = 200, quick: bool = False,
              on_progress=None):
    """Run the full algorithm audit."""

    def progress(msg):
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

    start_time = time.time()
    db = EmbeddingDatabase(db_path)
    corpus = load_corpus(db, on_progress=progress)
    db.close()

    n_tracks = len(corpus.track_ids)
    dim = corpus.embeddings.shape[1]
    progress(f"\nCorpus: {n_tracks} tracks, {dim}d embeddings")

    # Select seeds
    seeds = select_seeds(corpus, n_seeds)
    progress(f"Selected {len(seeds)} seed tracks across {n_seeds} clusters\n")

    # Storage
    all_metrics: dict[str, list[RunMetrics]] = defaultdict(list)
    validations: list[ValidationResult] = []
    post_filter_violations = 0
    post_filter_total = 0

    max_per_artist = 3
    min_spacing = 3

    # -----------------------------------------------------------------------
    # Validation 12: Full-corpus nearest-neighbor sanity check
    # -----------------------------------------------------------------------
    progress("=" * 60)
    progress("Validation 12: Nearest-neighbor sanity check")
    progress("=" * 60)

    rng = np.random.default_rng(42)
    check_indices = rng.choice(n_tracks, size=min(1000, n_tracks), replace=False)
    same_artist_in_top10 = 0
    checked = 0

    for idx in check_indices:
        tid = int(corpus.track_ids[idx])
        artist = corpus.artists.get(tid)
        if artist is None:
            continue

        neighbors = find_top_k(corpus, corpus.embeddings[idx], 11, exclude_ids={tid})
        neighbor_artists = [corpus.artists.get(n[0]) for n in neighbors[:10]]
        if artist in neighbor_artists:
            same_artist_in_top10 += 1
        checked += 1

    if checked > 0:
        rate = same_artist_in_top10 / checked
        progress(f"  Same-artist in top-10 neighbors: {same_artist_in_top10}/{checked} ({rate:.1%})")
        validations.append(ValidationResult(
            "NN sanity check",
            rate > 0.05,  # At least 5% should have same-artist neighbor
            f"Same-artist rate: {rate:.1%} ({same_artist_in_top10}/{checked})"
        ))

    # -----------------------------------------------------------------------
    # Validation 11: Decay schedule formulas (pure math, zero seeds needed)
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Validation 11: Decay schedule formula verification")
    progress("=" * 60)

    base_alpha = 0.5
    total = 50
    positions = [0, 12, 25, 37, 49]

    expected = {
        "none": [0.5, 0.5, 0.5, 0.5, 0.5],
        "linear": [base_alpha * (1 - p / 49) for p in positions],
        "exponential": [base_alpha * np.exp(-3 * p / 49) for p in positions],
        "step": [base_alpha if p / 49 < 0.5 else base_alpha * 0.2 for p in positions],
    }

    all_match = True
    for decay_name, expected_vals in expected.items():
        for i, pos in enumerate(positions):
            actual = compute_alpha(base_alpha, pos, total, decay_name)
            exp = expected_vals[i]
            if abs(actual - exp) > 1e-6:
                progress(f"  FAIL: {decay_name} at step {pos}: expected {exp:.6f}, got {actual:.6f}")
                all_match = False

    progress(f"  Decay formulas: {'PASS' if all_match else 'FAIL'}")
    validations.append(ValidationResult("Decay schedule formulas", all_match))

    # -----------------------------------------------------------------------
    # Validation 1: DPP+drift degeneracy proof
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Validation 1: DPP+drift = MMR(lambda=1)+drift degeneracy")
    progress("=" * 60)

    proof_seeds = seeds[:20]
    identical_count = 0
    for i, seed_tid in enumerate(proof_seeds):
        dpp_result = run_drift_pipeline(
            seed_tid, corpus, mode="dpp", drift_mode="seed_interpolation",
            anchor_strength=0.5, anchor_decay="exponential",
            num_tracks=50, max_per_artist=100, min_spacing=0  # No post-filter interference
        )
        mmr1_result = run_drift_pipeline(
            seed_tid, corpus, mode="mmr", drift_mode="seed_interpolation",
            lambda_=1.0, anchor_strength=0.5, anchor_decay="exponential",
            num_tracks=50, max_per_artist=100, min_spacing=0
        )

        dpp_tids = [t[0] for t in dpp_result]
        mmr1_tids = [t[0] for t in mmr1_result]

        if dpp_tids == mmr1_tids:
            identical_count += 1
        else:
            # Find first divergence
            for j in range(min(len(dpp_tids), len(mmr1_tids))):
                if dpp_tids[j] != mmr1_tids[j]:
                    progress(f"  Seed {i}: diverged at step {j} "
                             f"(DPP={dpp_tids[j]}, MMR1={mmr1_tids[j]})")
                    break

    progress(f"  Identical: {identical_count}/{len(proof_seeds)}")
    validations.append(ValidationResult(
        "DPP+drift = MMR(lambda=1)+drift",
        identical_count == len(proof_seeds),
        f"{identical_count}/{len(proof_seeds)} identical"
    ))

    # -----------------------------------------------------------------------
    # MMR batch mode (Validation 2: lambda monotonicity)
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("MMR batch mode: lambda sweep")
    progress("=" * 60)

    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mmr_by_lambda: dict[float, list[RunMetrics]] = defaultdict(list)

    for li, lam in enumerate(lambdas):
        for si, seed_tid in enumerate(seeds):
            result = run_batch_pipeline(
                seed_tid, corpus, mode="mmr", lambda_=lam,
                max_per_artist=max_per_artist, min_spacing=min_spacing
            )
            m = compute_metrics(seed_tid, result, corpus, mode="mmr",
                                config={"lambda": lam})
            mmr_by_lambda[lam].append(m)
            all_metrics[f"mmr_batch_l{lam}"].append(m)

            # Check post-filter
            post_filter_total += 1
            if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                post_filter_violations += 1

        avg_sim = np.mean([m.mean_sim_to_seed for m in mmr_by_lambda[lam]])
        avg_pw = np.mean([m.mean_pairwise_sim for m in mmr_by_lambda[lam]])
        avg_artists = np.mean([m.unique_artists for m in mmr_by_lambda[lam]])
        progress(f"  lambda={lam:.1f}: sim_to_seed={avg_sim:.4f}, "
                 f"pairwise={avg_pw:.4f}, artists={avg_artists:.1f}")

    # Validation 2: Monotonicity
    mono_sim_violations = 0
    mono_pw_violations = 0
    mono_art_violations = 0

    for si in range(len(seeds)):
        sims_by_lambda = [mmr_by_lambda[l][si].mean_sim_to_seed for l in lambdas]
        pw_by_lambda = [mmr_by_lambda[l][si].mean_pairwise_sim for l in lambdas]
        art_by_lambda = [mmr_by_lambda[l][si].unique_artists for l in lambdas]

        for j in range(1, len(lambdas)):
            if sims_by_lambda[j] < sims_by_lambda[j - 1] - 0.001:
                mono_sim_violations += 1
                break
        for j in range(1, len(lambdas)):
            if pw_by_lambda[j] < pw_by_lambda[j - 1] - 0.001:
                mono_pw_violations += 1
                break
        for j in range(1, len(lambdas)):
            if art_by_lambda[j] > art_by_lambda[j - 1] + 1:
                mono_art_violations += 1
                break

    pct = lambda v: v / len(seeds) * 100
    progress(f"\n  Monotonicity violations:")
    progress(f"    sim_to_seed non-increasing: {mono_sim_violations}/{len(seeds)} ({pct(mono_sim_violations):.1f}%)")
    progress(f"    pairwise non-increasing: {mono_pw_violations}/{len(seeds)} ({pct(mono_pw_violations):.1f}%)")
    progress(f"    artists non-decreasing: {mono_art_violations}/{len(seeds)} ({pct(mono_art_violations):.1f}%)")

    threshold_strict = 0.05 * len(seeds)
    # Artist monotonicity has wider tolerance because post-filter (maxPerArtist + spacing)
    # interacts with selection in non-monotonic ways: higher lambda concentrates picks on
    # a few top artists, post-filter drops them, potentially yielding MORE unique artists
    threshold_artist = 0.30 * len(seeds)
    validations.append(ValidationResult(
        "MMR lambda monotonicity (sim_to_seed)",
        mono_sim_violations <= threshold_strict,
        f"{mono_sim_violations}/{len(seeds)} violations (threshold: {threshold_strict:.0f})"
    ))
    validations.append(ValidationResult(
        "MMR lambda monotonicity (pairwise)",
        mono_pw_violations <= threshold_strict,
        f"{mono_pw_violations}/{len(seeds)} violations"
    ))
    validations.append(ValidationResult(
        "MMR lambda monotonicity (artists)",
        mono_art_violations <= threshold_artist,
        f"{mono_art_violations}/{len(seeds)} violations (tolerance 30% due to post-filter interaction)"
    ))

    # -----------------------------------------------------------------------
    # DPP batch mode
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("DPP batch mode")
    progress("=" * 60)

    dpp_metrics = []
    for si, seed_tid in enumerate(seeds):
        result = run_batch_pipeline(
            seed_tid, corpus, mode="dpp",
            max_per_artist=max_per_artist, min_spacing=min_spacing
        )
        m = compute_metrics(seed_tid, result, corpus, mode="dpp")
        dpp_metrics.append(m)
        all_metrics["dpp_batch"].append(m)

        post_filter_total += 1
        if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
            post_filter_violations += 1

    avg_sim = np.mean([m.mean_sim_to_seed for m in dpp_metrics])
    avg_pw = np.mean([m.mean_pairwise_sim for m in dpp_metrics])
    avg_artists = np.mean([m.unique_artists for m in dpp_metrics])
    progress(f"  sim_to_seed={avg_sim:.4f}, pairwise={avg_pw:.4f}, artists={avg_artists:.1f}")

    # -----------------------------------------------------------------------
    # Validation 5: DPP > MMR diversity
    # -----------------------------------------------------------------------
    mmr_default = mmr_by_lambda[0.6]
    dpp_better_count = 0
    for si in range(len(seeds)):
        if dpp_metrics[si].mean_pairwise_sim < mmr_default[si].mean_pairwise_sim:
            dpp_better_count += 1

    pct_better = dpp_better_count / len(seeds) * 100
    progress(f"\n  DPP < MMR(0.6) pairwise sim: {dpp_better_count}/{len(seeds)} ({pct_better:.1f}%)")
    validations.append(ValidationResult(
        "DPP > MMR diversity (pairwise_sim)",
        dpp_better_count >= 0.90 * len(seeds),
        f"DPP better in {pct_better:.1f}% of seeds"
    ))

    # -----------------------------------------------------------------------
    # Temperature batch mode (Validation 3 & 4)
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Temperature batch mode: tau sweep")
    progress("=" * 60)

    taus = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    n_runs = 10 if not quick else 3
    temp_by_tau: dict[float, list[list[RunMetrics]]] = defaultdict(list)  # tau -> [runs_per_seed]

    for ti, tau in enumerate(taus):
        for si, seed_tid in enumerate(seeds):
            seed_runs = []
            for run_idx in range(n_runs):
                run_rng = np.random.default_rng(42 + si * 1000 + run_idx)
                result = run_batch_pipeline(
                    seed_tid, corpus, mode="temperature", temperature=tau,
                    max_per_artist=max_per_artist, min_spacing=min_spacing,
                    rng=run_rng
                )
                m = compute_metrics(seed_tid, result, corpus, mode="temperature",
                                    config={"tau": tau, "run": run_idx})
                seed_runs.append(m)
                all_metrics[f"temp_batch_t{tau}"].append(m)

                post_filter_total += 1
                if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                    post_filter_violations += 1

            temp_by_tau[tau].append(seed_runs)

        # Average across seeds and runs
        all_sims = [m.mean_sim_to_seed for runs in temp_by_tau[tau] for m in runs]
        avg_sim = np.mean(all_sims)
        # Variance across runs per seed
        variances = []
        for runs in temp_by_tau[tau]:
            run_sims = [m.mean_sim_to_seed for m in runs]
            variances.append(np.std(run_sims))
        avg_var = np.mean(variances)
        progress(f"  tau={tau:.2f}: sim_to_seed={avg_sim:.4f}, run_std={avg_var:.4f}")

    # Validation 3: T~0 produces higher overlap than T=1
    # Note: True determinism (identical runs) requires T < 1e-6 (Kotlin threshold).
    # At T=0.01, Gumbel noise (-2..5) still matters vs score/0.01 spread (~3 units),
    # so we test that LOW temperature produces MORE overlap between runs than HIGH.
    if 0.01 in temp_by_tau and 1.0 in temp_by_tau:
        low_t_overlaps = []
        high_t_overlaps = []
        check_total = min(len(seeds), 20)
        for si in range(check_total):
            # Pairwise Jaccard overlap of top-10 across runs
            for tau_val, overlaps_list in [(0.01, low_t_overlaps), (1.0, high_t_overlaps)]:
                runs = temp_by_tau[tau_val][si]
                for a in range(len(runs)):
                    for b in range(a + 1, len(runs)):
                        set_a = set(runs[a].selected_tids[:10])
                        set_b = set(runs[b].selected_tids[:10])
                        if set_a or set_b:
                            overlaps_list.append(len(set_a & set_b) / len(set_a | set_b))

        avg_low = np.mean(low_t_overlaps) if low_t_overlaps else 0
        avg_high = np.mean(high_t_overlaps) if high_t_overlaps else 0
        progress(f"\n  T=0.01 avg run overlap (Jaccard): {avg_low:.3f}")
        progress(f"  T=1.00 avg run overlap (Jaccard): {avg_high:.3f}")
        validations.append(ValidationResult(
            "Temperature: low T more consistent than high T",
            avg_low >= avg_high,
            f"T=0.01 overlap={avg_low:.3f}, T=1.0 overlap={avg_high:.3f}"
        ))

    # Validation 4: Temperature scaling
    tau_avg_sims = {}
    tau_avg_vars = {}
    for tau in taus:
        sims = [m.mean_sim_to_seed for runs in temp_by_tau[tau] for m in runs]
        tau_avg_sims[tau] = np.mean(sims)
        vars_ = [np.std([m.mean_sim_to_seed for m in runs]) for runs in temp_by_tau[tau]]
        tau_avg_vars[tau] = np.mean(vars_)

    sim_decreasing = all(tau_avg_sims[taus[i]] >= tau_avg_sims[taus[i + 1]] - 0.01
                         for i in range(len(taus) - 1))

    # Measure set diversity: how different are track lists across runs at each temp?
    # Low T should produce high overlap (similar lists), high T should produce low overlap
    tau_avg_set_overlap = {}
    for tau in taus:
        overlaps = []
        for runs in temp_by_tau[tau]:
            for a in range(len(runs)):
                for b in range(a + 1, len(runs)):
                    set_a = set(runs[a].selected_tids)
                    set_b = set(runs[b].selected_tids)
                    if set_a or set_b:
                        overlaps.append(len(set_a & set_b) / len(set_a | set_b))
        tau_avg_set_overlap[tau] = np.mean(overlaps) if overlaps else 0

    overlap_decreasing = tau_avg_set_overlap[taus[0]] > tau_avg_set_overlap[taus[-1]]

    progress(f"  Sim decreasing with tau: {sim_decreasing}")
    progress(f"  Set overlap decreasing with tau: {overlap_decreasing}")
    progress(f"    overlaps: {[f'{tau_avg_set_overlap[t]:.3f}' for t in taus]}")
    validations.append(ValidationResult(
        "Temperature scaling (sim decreases)",
        sim_decreasing,
        f"sims: {[f'{tau_avg_sims[t]:.4f}' for t in taus]}"
    ))
    validations.append(ValidationResult(
        "Temperature scaling (set overlap decreases)",
        overlap_decreasing,
        f"overlaps: {[f'{tau_avg_set_overlap[t]:.3f}' for t in taus]}"
    ))

    # -----------------------------------------------------------------------
    # Random Walk (Validation 6)
    # -----------------------------------------------------------------------
    if corpus.graph_ids is not None:
        progress("\n" + "=" * 60)
        progress("Random Walk: alpha sweep")
        progress("=" * 60)

        rw_alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        rw_by_alpha: dict[float, list[RunMetrics]] = defaultdict(list)

        for alpha in rw_alphas:
            transitive_counts = []
            for si, seed_tid in enumerate(seeds):
                result = run_batch_pipeline(
                    seed_tid, corpus, mode="random_walk", alpha=alpha,
                    max_per_artist=max_per_artist, min_spacing=min_spacing
                )
                m = compute_metrics(seed_tid, result, corpus, mode="random_walk",
                                    config={"alpha": alpha})

                # Compute transitive count: tracks NOT in brute-force top-200
                seed_idx = corpus.tid_to_idx.get(seed_tid)
                if seed_idx is not None:
                    bf_top200 = find_top_k(corpus, corpus.embeddings[seed_idx], 200,
                                           exclude_ids={seed_tid})
                    bf_tids = {t[0] for t in bf_top200}
                    m.transitive_count = sum(1 for t, _ in result if t not in bf_tids)
                    transitive_counts.append(m.transitive_count)

                rw_by_alpha[alpha].append(m)
                all_metrics[f"rw_a{alpha}"].append(m)

                post_filter_total += 1
                if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                    post_filter_violations += 1

            avg_trans = np.mean(transitive_counts) if transitive_counts else 0
            avg_sim = np.mean([m.mean_sim_to_seed for m in rw_by_alpha[alpha]])
            progress(f"  alpha={alpha:.1f}: sim_to_seed={avg_sim:.4f}, "
                     f"transitive={avg_trans:.1f}")

        # Validation 6: Transitive discovery
        low_alpha_trans = np.mean([m.transitive_count for m in rw_by_alpha[0.1]])
        high_alpha_trans = np.mean([m.transitive_count for m in rw_by_alpha[0.9]])
        progress(f"\n  Transitive: alpha=0.1 avg={low_alpha_trans:.1f}, alpha=0.9 avg={high_alpha_trans:.1f}")
        validations.append(ValidationResult(
            "Random Walk transitive discovery",
            low_alpha_trans > high_alpha_trans,
            f"alpha=0.1: {low_alpha_trans:.1f}, alpha=0.9: {high_alpha_trans:.1f}"
        ))

    # -----------------------------------------------------------------------
    # Drift mode: MMR + seed interpolation (Validation 7 & 10)
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("MMR + Drift (seed interpolation): lambda x alpha x decay sweep")
    progress("=" * 60)

    drift_lambdas = [0.4, 0.6, 0.8] if not quick else [0.6]
    drift_alphas = [0.2, 0.5, 0.8] if not quick else [0.5]
    drift_decays = ["none", "linear", "exponential", "step"] if not quick else ["none", "exponential"]

    drift_si_metrics: dict[str, list[RunMetrics]] = defaultdict(list)

    for lam in drift_lambdas:
        for alpha in drift_alphas:
            for decay in drift_decays:
                key = f"mmr_drift_si_l{lam}_a{alpha}_{decay}"
                for si, seed_tid in enumerate(seeds):
                    result = run_drift_pipeline(
                        seed_tid, corpus, mode="mmr", drift_mode="seed_interpolation",
                        lambda_=lam, anchor_strength=alpha, anchor_decay=decay,
                        max_per_artist=max_per_artist, min_spacing=min_spacing
                    )
                    m = compute_metrics(seed_tid, result, corpus, mode="mmr_drift_si",
                                        config={"lambda": lam, "alpha": alpha, "decay": decay})
                    drift_si_metrics[key].append(m)
                    all_metrics[key].append(m)

                    post_filter_total += 1
                    if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                        post_filter_violations += 1

                avg_sim = np.mean([m.mean_sim_to_seed for m in drift_si_metrics[key]])
                progress(f"  l={lam} a={alpha} {decay:12s}: sim_to_seed={avg_sim:.4f}")

    # Validation 7: Drift sim decay curves
    progress("\n  Drift sim decay curves:")
    decay_valid = True
    for decay in drift_decays:
        key = f"mmr_drift_si_l0.6_a0.5_{decay}"
        if key not in drift_si_metrics:
            continue
        metrics = drift_si_metrics[key]
        # Average sim at first and last position
        first_sims = [m.sim_at_position[0] for m in metrics if len(m.sim_at_position) > 0]
        last_sims = [m.sim_at_position[-1] for m in metrics if len(m.sim_at_position) > 10]
        if first_sims and last_sims:
            avg_first = np.mean(first_sims)
            avg_last = np.mean(last_sims)
            drops = avg_first > avg_last
            progress(f"    {decay:12s}: first={avg_first:.4f}, last={avg_last:.4f}, drops={drops}")
            if not drops:
                decay_valid = False

    validations.append(ValidationResult(
        "Drift sim decay (first > last)",
        decay_valid,
        "All decay schedules show decreasing sim"
    ))

    # Validation 10: Alpha != Lambda orthogonality
    progress("\n  Alpha vs Lambda orthogonality:")
    # Fix alpha=0.5, sweep lambda -> measure sim_to_seed
    lambda_curve = []
    for lam in drift_lambdas:
        key = f"mmr_drift_si_l{lam}_a0.5_exponential"
        if key in drift_si_metrics:
            lambda_curve.append(np.mean([m.mean_sim_to_seed for m in drift_si_metrics[key]]))

    # Fix lambda=0.6, sweep alpha -> measure sim_to_seed
    alpha_curve = []
    for alpha in drift_alphas:
        key = f"mmr_drift_si_l0.6_a{alpha}_exponential"
        if key in drift_si_metrics:
            alpha_curve.append(np.mean([m.mean_sim_to_seed for m in drift_si_metrics[key]]))

    if len(lambda_curve) >= 2 and len(alpha_curve) >= 2:
        lambda_range = max(lambda_curve) - min(lambda_curve)
        alpha_range = max(alpha_curve) - min(alpha_curve)
        progress(f"    Lambda sweep (alpha=0.5): range={lambda_range:.4f}, values={[f'{v:.4f}' for v in lambda_curve]}")
        progress(f"    Alpha sweep (lambda=0.6): range={alpha_range:.4f}, values={[f'{v:.4f}' for v in alpha_curve]}")

        both_affect = lambda_range > 0.005 and alpha_range > 0.005
        validations.append(ValidationResult(
            "Alpha != Lambda orthogonality",
            both_affect,
            f"lambda_range={lambda_range:.4f}, alpha_range={alpha_range:.4f}"
        ))

    # -----------------------------------------------------------------------
    # Drift mode: MMR + momentum (Validation 8)
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("MMR + Drift (momentum): lambda x beta sweep")
    progress("=" * 60)

    drift_betas = [0.3, 0.5, 0.7, 0.9] if not quick else [0.5, 0.9]
    mom_metrics: dict[str, list[RunMetrics]] = defaultdict(list)

    for lam in drift_lambdas:
        for beta in drift_betas:
            key = f"mmr_drift_mom_l{lam}_b{beta}"
            for si, seed_tid in enumerate(seeds):
                result = run_drift_pipeline(
                    seed_tid, corpus, mode="mmr", drift_mode="momentum",
                    lambda_=lam, momentum_beta=beta,
                    max_per_artist=max_per_artist, min_spacing=min_spacing
                )
                m = compute_metrics(seed_tid, result, corpus, mode="mmr_drift_mom",
                                    config={"lambda": lam, "beta": beta})
                mom_metrics[key].append(m)
                all_metrics[key].append(m)

                post_filter_total += 1
                if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                    post_filter_violations += 1

            avg_sim = np.mean([m.mean_sim_to_seed for m in mom_metrics[key]])
            avg_consec = np.mean([m.consecutive_sim for m in mom_metrics[key]])
            progress(f"  l={lam} beta={beta}: sim_to_seed={avg_sim:.4f}, consec_sim={avg_consec:.4f}")

    # Validation 8: Momentum vs seed interpolation smoothness
    progress("\n  Momentum vs Seed Interp trajectory smoothness:")
    mom_key = "mmr_drift_mom_l0.6_b0.7"
    si_key = "mmr_drift_si_l0.6_a0.5_exponential"
    if mom_key in mom_metrics and si_key in drift_si_metrics:
        mom_consec = np.mean([m.consecutive_sim for m in mom_metrics[mom_key]])
        si_consec = np.mean([m.consecutive_sim for m in drift_si_metrics[si_key]])
        progress(f"    Momentum consec_sim: {mom_consec:.4f}")
        progress(f"    Seed interp consec_sim: {si_consec:.4f}")
        validations.append(ValidationResult(
            "Momentum smoother than seed interpolation",
            mom_consec > si_consec,
            f"momentum={mom_consec:.4f}, seed_interp={si_consec:.4f}"
        ))

    # -----------------------------------------------------------------------
    # Drift mode: Temperature + seed interpolation
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Temperature + Drift (seed interpolation)")
    progress("=" * 60)

    temp_drift_taus = [0.05, 0.1, 0.3] if not quick else [0.1]
    temp_drift_alphas = [0.3, 0.5, 0.8] if not quick else [0.5]
    temp_drift_decays = ["none", "exponential"] if not quick else ["exponential"]
    temp_drift_runs = 5 if not quick else 2

    for tau in temp_drift_taus:
        for alpha in temp_drift_alphas:
            for decay in temp_drift_decays:
                key = f"temp_drift_si_t{tau}_a{alpha}_{decay}"
                for si, seed_tid in enumerate(seeds):
                    for run_idx in range(temp_drift_runs):
                        run_rng = np.random.default_rng(42 + si * 1000 + run_idx)
                        result = run_drift_pipeline(
                            seed_tid, corpus, mode="temperature", drift_mode="seed_interpolation",
                            temperature=tau, anchor_strength=alpha, anchor_decay=decay,
                            max_per_artist=max_per_artist, min_spacing=min_spacing,
                            rng=run_rng
                        )
                        m = compute_metrics(seed_tid, result, corpus, mode="temp_drift_si",
                                            config={"tau": tau, "alpha": alpha, "decay": decay})
                        all_metrics[key].append(m)

                        post_filter_total += 1
                        if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                            post_filter_violations += 1

                avg_sim = np.mean([m.mean_sim_to_seed for m in all_metrics[key]])
                progress(f"  tau={tau} a={alpha} {decay:12s}: sim_to_seed={avg_sim:.4f}")

    # -----------------------------------------------------------------------
    # Drift mode: Temperature + momentum
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Temperature + Drift (momentum)")
    progress("=" * 60)

    temp_mom_betas = [0.3, 0.7] if not quick else [0.7]

    for tau in temp_drift_taus:
        for beta in temp_mom_betas:
            key = f"temp_drift_mom_t{tau}_b{beta}"
            for si, seed_tid in enumerate(seeds):
                for run_idx in range(temp_drift_runs):
                    run_rng = np.random.default_rng(42 + si * 1000 + run_idx)
                    result = run_drift_pipeline(
                        seed_tid, corpus, mode="temperature", drift_mode="momentum",
                        temperature=tau, momentum_beta=beta,
                        max_per_artist=max_per_artist, min_spacing=min_spacing,
                        rng=run_rng
                    )
                    m = compute_metrics(seed_tid, result, corpus, mode="temp_drift_mom",
                                        config={"tau": tau, "beta": beta})
                    all_metrics[key].append(m)

                    post_filter_total += 1
                    if not validate_post_filter(result, corpus, max_per_artist, min_spacing):
                        post_filter_violations += 1

            avg_sim = np.mean([m.mean_sim_to_seed for m in all_metrics[key]])
            progress(f"  tau={tau} beta={beta}: sim_to_seed={avg_sim:.4f}")

    # -----------------------------------------------------------------------
    # Validation 9: Post-filter correctness
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("Validation 9: Post-filter correctness")
    progress("=" * 60)
    progress(f"  Checked {post_filter_total} runs, violations: {post_filter_violations}")
    validations.append(ValidationResult(
        "Post-filter correctness",
        post_filter_violations == 0,
        f"0/{post_filter_total} violations" if post_filter_violations == 0
        else f"{post_filter_violations}/{post_filter_total} violations"
    ))

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time
    progress("\n" + "=" * 60)
    progress("AUDIT RESULTS")
    progress("=" * 60)
    progress(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    progress(f"Seeds: {len(seeds)}, Total runs: {post_filter_total}")
    progress("")

    # Summary table
    max_name_len = max(len(v.name) for v in validations)
    passed_count = sum(1 for v in validations if v.passed)
    for v in validations:
        status = "PASS" if v.passed else "FAIL"
        progress(f"  [{status}] {v.name:<{max_name_len}}  {v.detail}")

    progress(f"\n  {passed_count}/{len(validations)} validations passed")

    # -----------------------------------------------------------------------
    # Metrics summary table
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("METRICS SUMMARY (mean +/- std across seeds)")
    progress("=" * 60)

    header = f"  {'Config':<45} {'sim_seed':>10} {'pairwise':>10} {'consec':>10} {'artists':>8} {'HHI':>8} {'len':>5}"
    progress(header)
    progress("  " + "-" * (len(header) - 2))

    for config_key in sorted(all_metrics.keys()):
        metrics_list = all_metrics[config_key]
        sim_seeds = [m.mean_sim_to_seed for m in metrics_list]
        pairwise = [m.mean_pairwise_sim for m in metrics_list]
        consec = [m.consecutive_sim for m in metrics_list]
        artists_ = [m.unique_artists for m in metrics_list]
        hhis = [m.artist_hhi for m in metrics_list]
        lens = [m.queue_length for m in metrics_list]

        progress(f"  {config_key:<45} "
                 f"{np.mean(sim_seeds):>10.4f} "
                 f"{np.mean(pairwise):>10.4f} "
                 f"{np.mean(consec):>10.4f} "
                 f"{np.mean(artists_):>8.1f} "
                 f"{np.mean(hhis):>8.4f} "
                 f"{np.mean(lens):>5.1f}")

    # -----------------------------------------------------------------------
    # Recommendations
    # -----------------------------------------------------------------------
    progress("\n" + "=" * 60)
    progress("RECOMMENDATIONS")
    progress("=" * 60)

    # Check DPP+drift degeneracy
    dpp_drift_v = next((v for v in validations if "DPP+drift" in v.name), None)
    if dpp_drift_v and dpp_drift_v.passed:
        progress("  [CONFIRMED] DPP+drift is degenerate (= MMR lambda=1 drift)")
        progress("    -> Disable drift for DPP mode in Android UI and engine")

    # Check if Temperature+drift is meaningfully different from Temperature batch
    progress("")
    if "temp_drift_si_t0.1_a0.5_exponential" in all_metrics and "temp_batch_t0.1" in all_metrics:
        drift_sim = np.mean([m.mean_sim_to_seed for m in all_metrics["temp_drift_si_t0.1_a0.5_exponential"]])
        batch_sim = np.mean([m.mean_sim_to_seed for m in all_metrics["temp_batch_t0.1"]])
        diff = abs(drift_sim - batch_sim)
        progress(f"  Temperature drift vs batch (tau=0.1): drift_sim={drift_sim:.4f}, batch_sim={batch_sim:.4f}, diff={diff:.4f}")
        if diff < 0.01:
            progress("    -> Temperature+drift produces similar results to batch — consider simplifying")
        else:
            progress("    -> Temperature+drift is meaningfully different from batch — keep")

    # DPP diversity advantage
    dpp_v = next((v for v in validations if "DPP > MMR" in v.name), None)
    if dpp_v and dpp_v.passed:
        progress("  [CONFIRMED] DPP produces better diversity than MMR(0.6) in batch mode")

    # Random Walk transitive
    rw_v = next((v for v in validations if "transitive" in v.name), None)
    if rw_v and rw_v.passed:
        progress("  [CONFIRMED] Random Walk discovers transitive connections")

    # Write JSON results
    json_path = Path(str(db_path).replace('.db', '_audit.json'))
    progress(f"\n  Writing detailed results to {json_path}")

    json_data = {
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "n_tracks": n_tracks,
        "dim": dim,
        "total_runs": post_filter_total,
        "validations": [
            {"name": v.name, "passed": v.passed, "detail": v.detail}
            for v in validations
        ],
        "metrics_summary": {}
    }

    for config_key, metrics_list in sorted(all_metrics.items()):
        sim_seeds = [m.mean_sim_to_seed for m in metrics_list]
        pairwise = [m.mean_pairwise_sim for m in metrics_list]
        consec = [m.consecutive_sim for m in metrics_list]
        artists_ = [m.unique_artists for m in metrics_list]

        json_data["metrics_summary"][config_key] = {
            "n_runs": len(metrics_list),
            "mean_sim_to_seed": float(np.mean(sim_seeds)),
            "std_sim_to_seed": float(np.std(sim_seeds)),
            "mean_pairwise_sim": float(np.mean(pairwise)),
            "std_pairwise_sim": float(np.std(pairwise)),
            "mean_consecutive_sim": float(np.mean(consec)),
            "mean_unique_artists": float(np.mean(artists_)),
            "mean_queue_length": float(np.mean([m.queue_length for m in metrics_list])),
        }

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)

    progress(f"\nAudit complete: {passed_count}/{len(validations)} passed")

    return validations, all_metrics
