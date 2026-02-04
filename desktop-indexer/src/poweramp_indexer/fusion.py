"""Embedding fusion: SVD projection, k-means clustering, and kNN graph construction."""

import logging
import struct
from pathlib import Path

import numpy as np
from scipy.linalg import svd

from .database import EmbeddingDatabase, float_list_to_blob

logger = logging.getLogger(__name__)


def fuse_embeddings(db: EmbeddingDatabase, target_dim: int = 512, n_clusters: int = 200,
                    knn_k: int = 20, graph_path: Path | None = None,
                    on_progress=None):
    """
    Fuse MuLan + Flamingo embeddings via SVD, compute clusters and kNN graph.

    Steps:
    1. Load both model embeddings, zero-pad tracks with only one model
    2. Concatenate into 1024-dim space (equal weights)
    3. SVD project to target_dim
    4. L2-normalize
    5. Store in embeddings_fused table
    6. k-means clustering (K=n_clusters)
    7. kNN graph (K=knn_k) stored as binary file

    Args:
        db: Database with mulan and/or flamingo embeddings
        target_dim: Output embedding dimension (default 512)
        n_clusters: Number of k-means clusters (default 200)
        knn_k: Number of nearest neighbors for graph (default 20)
        graph_path: Where to write graph.bin (default: next to database)
        on_progress: Callback(message) for status updates
    """
    def progress(msg):
        logger.info(msg)
        if on_progress:
            on_progress(msg)

    # --- Step 1: Load embeddings ---
    progress("Loading embeddings...")

    mulan_embs = db.get_all_embeddings(model="mulan")
    flamingo_embs = db.get_all_embeddings(model="flamingo")

    if not mulan_embs and not flamingo_embs:
        raise ValueError("Database has neither MuLan nor Flamingo embeddings")

    # Detect dimensions
    mulan_dim = len(next(iter(mulan_embs.values()))) if mulan_embs else 512
    flamingo_dim = len(next(iter(flamingo_embs.values()))) if flamingo_embs else 512

    if mulan_dim != flamingo_dim:
        raise ValueError(
            f"MuLan dim ({mulan_dim}) != Flamingo dim ({flamingo_dim}). "
            f"Run 'reduce' on Flamingo first to match dimensions."
        )

    source_dim = mulan_dim  # Both should be same dim (e.g. 512 after reduction)
    concat_dim = source_dim * 2

    if target_dim > concat_dim:
        raise ValueError(f"Target dim ({target_dim}) > concatenated dim ({concat_dim})")

    # Collect all track IDs from both models
    all_track_ids = sorted(set(mulan_embs.keys()) | set(flamingo_embs.keys()))
    n_tracks = len(all_track_ids)
    both_count = len(set(mulan_embs.keys()) & set(flamingo_embs.keys()))

    progress(f"Tracks: {n_tracks} total ({both_count} with both models, "
             f"{len(mulan_embs) - both_count} MuLan-only, "
             f"{len(flamingo_embs) - both_count} Flamingo-only)")

    # --- Step 2: Concatenate with zero-padding for missing models ---
    progress(f"Building {n_tracks} x {concat_dim} concatenated matrix...")

    track_ids = np.array(all_track_ids, dtype=np.int64)
    X = np.zeros((n_tracks, concat_dim), dtype=np.float32)

    zero_mulan = [0.0] * source_dim
    zero_flamingo = [0.0] * source_dim

    for i, tid in enumerate(all_track_ids):
        mulan_vec = mulan_embs.get(tid, zero_mulan)
        flamingo_vec = flamingo_embs.get(tid, zero_flamingo)
        X[i, :source_dim] = mulan_vec
        X[i, source_dim:] = flamingo_vec

    # Free memory
    del mulan_embs, flamingo_embs

    # --- Step 3: SVD projection ---
    progress(f"Computing SVD ({n_tracks} x {concat_dim} -> {target_dim})...")

    # Full SVD on the data matrix
    U, s, Vt = svd(X.astype(np.float64), full_matrices=False)

    total_var = np.sum(s ** 2)
    retained_var = np.sum(s[:target_dim] ** 2) / total_var
    progress(f"Variance retained: {retained_var * 100:.2f}%")

    # Project: X_reduced = U[:, :target_dim] * s[:target_dim]
    X_reduced = (U[:, :target_dim] * s[:target_dim]).astype(np.float32)

    # Save projection matrix for on-device use: Vt[:target_dim, :] (target_dim x concat_dim)
    projection_matrix = Vt[:target_dim, :].astype(np.float32)

    del U, Vt, X  # Free memory

    # --- Step 4: L2-normalize ---
    progress("L2-normalizing...")
    norms = np.linalg.norm(X_reduced, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    X_reduced = X_reduced / norms

    # --- Step 5: Store fused embeddings ---
    progress(f"Writing {n_tracks} fused embeddings to database...")

    # Ensure table exists
    db._init_schema(["fused"])

    db.conn.execute("BEGIN")
    try:
        for i in range(n_tracks):
            blob = float_list_to_blob(X_reduced[i].tolist())
            db.conn.execute(
                "INSERT OR REPLACE INTO embeddings_fused (track_id, embedding) VALUES (?, ?)",
                (int(track_ids[i]), blob)
            )
            if (i + 1) % 10000 == 0:
                progress(f"  written {i + 1}/{n_tracks} embeddings")
        db.conn.execute("COMMIT")
    except Exception:
        db.conn.execute("ROLLBACK")
        raise

    progress(f"  written {n_tracks}/{n_tracks} embeddings")

    # Store projection matrix and metadata
    db.conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("fused_projection", projection_matrix.tobytes())
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("fused_dim", str(target_dim))
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("fused_source_dim", str(concat_dim))
    )
    db.conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("fused_variance_retained", f"{retained_var:.6f}")
    )
    db.conn.commit()

    # --- Step 6: k-means clustering ---
    progress(f"Running k-means (K={n_clusters})...")
    labels, centroids = _kmeans(X_reduced, n_clusters, max_iter=100, on_progress=progress)

    # Store cluster assignments
    progress("Writing cluster assignments...")

    # Add cluster_id column if not exists
    try:
        db.conn.execute("ALTER TABLE tracks ADD COLUMN cluster_id INTEGER")
    except Exception:
        pass  # Column already exists

    # Create clusters table
    db.conn.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id INTEGER PRIMARY KEY,
            embedding BLOB NOT NULL
        )
    """)

    db.conn.execute("BEGIN")
    try:
        # Store assignments
        for i in range(n_tracks):
            db.conn.execute(
                "UPDATE tracks SET cluster_id = ? WHERE id = ?",
                (int(labels[i]), int(track_ids[i]))
            )

        # Store centroids
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

    # --- Step 7: kNN graph ---
    progress(f"Building kNN graph (K={knn_k})...")
    neighbors, weights = _build_knn_graph(X_reduced, track_ids, knn_k, on_progress=progress)

    # Write graph.bin
    if graph_path is None:
        graph_path = db.db_path.parent / "graph.bin"

    _write_graph_binary(graph_path, track_ids, neighbors, weights, knn_k)

    file_size_mb = graph_path.stat().st_size / 1024 / 1024
    progress(f"Wrote graph.bin: {file_size_mb:.1f} MB")

    return {
        "n_tracks": n_tracks,
        "target_dim": target_dim,
        "variance_retained": retained_var,
        "n_clusters": n_clusters,
        "knn_k": knn_k,
        "graph_path": str(graph_path),
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

    # First centroid: random
    centroids[0] = X[rng.integers(n)]

    # Remaining centroids: probabilistic furthest-first
    for i in range(1, k):
        # Cosine similarities to nearest centroid
        sims = X @ centroids[:i].T  # (N, i)
        max_sim = sims.max(axis=1)  # Nearest centroid similarity
        # Distance = 1 - sim; probability proportional to distance²
        dists = np.maximum(1.0 - max_sim, 0.0)
        probs = dists ** 2
        probs_sum = probs.sum()
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(n) / n
        centroids[i] = X[rng.choice(n, p=probs)]

    # Iterate
    labels = np.zeros(n, dtype=np.int32)
    for iteration in range(max_iter):
        # Assign: cosine similarity = dot product for unit vectors
        sims = X @ centroids.T  # (N, K)
        new_labels = sims.argmax(axis=1)

        changed = (new_labels != labels).sum()
        labels = new_labels

        if on_progress and (iteration % 10 == 0 or changed == 0):
            on_progress(f"  k-means iter {iteration}: {changed} reassignments")

        if changed == 0:
            break

        # Update centroids
        for j in range(k):
            mask = labels == j
            if mask.any():
                centroids[j] = X[mask].mean(axis=0)
                # Re-normalize to unit sphere
                norm = np.linalg.norm(centroids[j])
                if norm > 1e-10:
                    centroids[j] /= norm

    return labels, centroids


def _build_knn_graph(X: np.ndarray, track_ids: np.ndarray, k: int,
                     on_progress=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Build kNN graph with row-normalized edge weights (transition probabilities).

    For each track, find K nearest neighbors by cosine similarity (dot product),
    then normalize weights so they sum to 1 per row.

    Args:
        X: Embeddings matrix (N, D), L2-normalized
        track_ids: Array of track IDs corresponding to rows of X
        k: Number of neighbors per node
        on_progress: Status callback

    Returns:
        (neighbors, weights) — both shape (N, K)
        neighbors[i] contains indices into track_ids array (not track IDs themselves)
        weights[i] are row-normalized transition probabilities
    """
    n = X.shape[0]
    neighbors = np.empty((n, k), dtype=np.int32)
    weights = np.empty((n, k), dtype=np.float32)

    # Process in chunks to avoid N×N memory
    chunk_size = 1000
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        # Similarities for this chunk against all embeddings
        sims = X[start:end] @ X.T  # (chunk, N)

        for i in range(start, end):
            row = sims[i - start]
            row[i] = -float('inf')  # Exclude self

            # Top-K indices
            top_k_idx = np.argpartition(row, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(row[top_k_idx])[::-1]]

            # Similarities as weights
            top_k_sims = np.maximum(row[top_k_idx], 0.0)  # Clamp negatives

            # Row-normalize to transition probabilities
            total = top_k_sims.sum()
            if total > 0:
                top_k_sims /= total

            neighbors[i] = top_k_idx
            weights[i] = top_k_sims

        if on_progress:
            on_progress(f"  kNN: {min(end, n)}/{n}")

    return neighbors, weights


def _write_graph_binary(path: Path, track_ids: np.ndarray,
                        neighbors: np.ndarray, weights: np.ndarray, k: int):
    """
    Write kNN graph as binary file.

    Format:
        Header: N (uint32), K (uint32)
        ID map: track_ids[N] (int64, little-endian) — maps index to track ID
        Graph: N * K entries of (neighbor_index uint32, weight float32)

    The neighbor_index values are indices into the ID map (not track IDs directly),
    keeping the graph compact.
    """
    n = len(track_ids)

    with open(path, 'wb') as f:
        # Header
        f.write(struct.pack('<II', n, k))

        # ID map
        for tid in track_ids:
            f.write(struct.pack('<q', int(tid)))

        # Graph data
        for i in range(n):
            for j in range(k):
                f.write(struct.pack('<If', int(neighbors[i, j]), float(weights[i, j])))
