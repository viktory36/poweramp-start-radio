"""Post-processing: k-means clustering and kNN graph construction for CLaMP3 embeddings."""

import logging
import struct

import numpy as np

from .database import EmbeddingDatabase, float_list_to_blob

logger = logging.getLogger(__name__)


def build_index(db: EmbeddingDatabase, n_clusters: int = 200,
                knn_k: int = 5, on_progress=None):
    """
    Build k-means clusters and kNN graph from CLaMP3 embeddings.

    Steps:
    1. Load all 768d embeddings
    2. k-means clustering (K=n_clusters)
    3. kNN graph (K=knn_k) stored as binary blob

    Args:
        db: Database with CLaMP3 embeddings
        n_clusters: Number of k-means clusters (default 200)
        knn_k: Number of nearest neighbors for graph (default 20)
        on_progress: Callback(message) for status updates
    """
    def progress(msg):
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    # --- Step 1: Load embeddings ---
    progress("Loading CLaMP3 embeddings...")

    embs = db.get_all_embeddings()
    if not embs:
        raise ValueError("Database has no CLaMP3 embeddings")

    all_track_ids = sorted(embs.keys())
    n_tracks = len(all_track_ids)
    dim = len(next(iter(embs.values())))

    progress(f"Loaded {n_tracks} embeddings ({dim}d)")

    track_ids = np.array(all_track_ids, dtype=np.int64)
    X = np.zeros((n_tracks, dim), dtype=np.float32)
    for i, tid in enumerate(all_track_ids):
        X[i] = embs[tid]

    del embs  # Free memory

    # Ensure L2-normalized
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X = X / norms

    # Store metadata
    db.set_metadata("model", "clamp3")
    db.set_metadata("embedding_dim", str(dim))

    # --- Step 2: k-means clustering ---
    progress(f"Running k-means (K={n_clusters})...")
    labels, centroids = _kmeans(X, n_clusters, max_iter=100, on_progress=progress)

    progress("Writing cluster assignments...")

    # Add cluster_id column if not exists
    try:
        db.conn.execute("ALTER TABLE tracks ADD COLUMN cluster_id INTEGER")
    except Exception:
        pass

    db.conn.execute("BEGIN")
    try:
        for i in range(n_tracks):
            db.conn.execute(
                "UPDATE tracks SET cluster_id = ? WHERE id = ?",
                (int(labels[i]), int(track_ids[i]))
            )

        db.conn.execute("DELETE FROM clusters")
        for k in range(n_clusters):
            blob = float_list_to_blob(centroids[k].tolist())
            db.conn.execute(
                "INSERT INTO clusters (cluster_id, embedding) VALUES (?, ?)",
                (k, blob)
            )

        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise

    progress(f"Stored {n_clusters} cluster centroids and {n_tracks} assignments")

    # --- Step 3: kNN graph ---
    progress(f"Building kNN graph (K={knn_k})...")
    neighbors, weights = _build_knn_graph(X, track_ids, knn_k, on_progress=progress)

    graph_blob = _build_graph_binary(track_ids, neighbors, weights, knn_k)
    db.set_binary("knn_graph", graph_blob)

    graph_size_mb = len(graph_blob) / 1024 / 1024
    progress(f"Stored kNN graph in database: {graph_size_mb:.1f} MB")

    return {
        "n_tracks": n_tracks,
        "embedding_dim": dim,
        "n_clusters": n_clusters,
        "knn_k": knn_k,
        "graph_size_mb": graph_size_mb,
    }


def _kmeans(X: np.ndarray, k: int, max_iter: int = 100,
            on_progress=None) -> tuple[np.ndarray, np.ndarray]:
    """
    k-means clustering on L2-normalized embeddings.

    Uses cosine distance (1 - dot product) since embeddings are unit-normalized.
    Centroids are re-normalized after each update to stay on the unit sphere.

    Returns:
        (labels, centroids) — labels shape (N,), centroids shape (K, D)
    """
    n, d = X.shape

    # Initialize with k-means++ on cosine distance
    rng = np.random.default_rng(42)
    centroids = np.empty((k, d), dtype=np.float32)

    centroids[0] = X[rng.integers(n)]

    for i in range(1, k):
        sims = X @ centroids[:i].T
        max_sim = sims.max(axis=1)
        dists = np.maximum(1.0 - max_sim, 0.0)
        probs = dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(n) / n
        centroids[i] = X[rng.choice(n, p=probs)]

    labels = np.zeros(n, dtype=np.int32)
    for iteration in range(max_iter):
        sims = X @ centroids.T
        new_labels = sims.argmax(axis=1)

        changed = (new_labels != labels).sum()
        labels = new_labels

        if on_progress and (iteration % 10 == 0 or changed == 0):
            on_progress(f"  k-means iter {iteration}: {changed} reassignments")

        if changed == 0:
            break

        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
                norm = np.linalg.norm(centroids[j])
                if norm > 1e-10:
                    centroids[j] /= norm

    return labels, centroids


def _build_knn_graph(X: np.ndarray, track_ids: np.ndarray, k: int,
                     on_progress=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph with row-normalized edge weights (transition probabilities).
    """
    n = X.shape[0]
    neighbors = np.empty((n, k), dtype=np.int32)
    weights = np.empty((n, k), dtype=np.float32)

    chunk_size = 1000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = X[start:end] @ X.T

        for i in range(start, end):
            row = sims[i - start]
            row[i] = -float('inf')

            top_k_idx = np.argpartition(row, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]

            top_k_sims = np.maximum(row[top_k_idx], 0.0)

            total = top_k_sims.sum()
            if total > 0:
                top_k_sims /= total

            neighbors[i] = top_k_idx
            weights[i] = top_k_sims

        if on_progress:
            on_progress(f"  kNN: {min(end, n)}/{n}")

    return neighbors, weights


def _build_graph_binary(track_ids: np.ndarray,
                        neighbors: np.ndarray, weights: np.ndarray, k: int) -> bytes:
    """
    Build kNN graph as binary blob.

    Format:
        Header: N (uint32), K (uint32)
        ID map: track_ids[N] (int64) — maps index to track ID
        Graph: N * K entries of (neighbor_index uint32, weight float32)
    """
    n = len(track_ids)
    parts = []

    parts.append(struct.pack('<II', n, k))

    for tid in track_ids:
        parts.append(struct.pack('<q', int(tid)))

    for i in range(n):
        for j in range(k):
            parts.append(struct.pack('<If', int(neighbors[i, j]), float(weights[i, j])))

    return b''.join(parts)
