"""Provenance math verification and ASCII rail diagram visualization.

Computes the theoretical provenance (influence weights) for each drift step,
runs brute-force cosine search to get actual tracks, and validates that
influence weights sum to ~1.0 and match the alpha/beta/decay formulas.

Usage:
    poweramp-indexer provenance fused.db [--seed "artist title"] [--random]
        [--mode mmr|temperature] [--drift interp|ema]
        [--alpha 0.4] [--beta 0.7] [--decay none|linear|exp|step]
        [--tracks 20]
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .database import EmbeddingDatabase


# ---------------------------------------------------------------------------
# Data loading (minimal, reused from audit pattern)
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceCorpus:
    """Minimal data for provenance verification."""
    embeddings: np.ndarray       # (N, D) float32, L2-normalized
    track_ids: np.ndarray        # (N,) int64
    artists: dict[int, str | None]
    titles: dict[int, str | None]
    tid_to_idx: dict[int, int]
    idx_to_tid: dict[int, int]


def load_corpus(db: EmbeddingDatabase, on_progress=None) -> ProvenanceCorpus:
    """Load fused embeddings and track metadata."""
    def progress(msg):
        if on_progress:
            on_progress(msg)

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

    # Load metadata
    progress("Loading track metadata...")
    artists: dict[int, str | None] = {}
    titles: dict[int, str | None] = {}
    meta_rows = db.conn.execute("SELECT id, artist, title FROM tracks").fetchall()
    for row in meta_rows:
        tid = row["id"]
        artists[tid] = row["artist"].lower().strip() if row["artist"] else None
        titles[tid] = row["title"]

    return ProvenanceCorpus(
        embeddings=embeddings, track_ids=track_ids,
        artists=artists, titles=titles,
        tid_to_idx=tid_to_idx, idx_to_tid=idx_to_tid,
    )


# ---------------------------------------------------------------------------
# Drift math — ported from DriftEngine.kt + RecommendationEngine.kt
# ---------------------------------------------------------------------------

def compute_alpha(base_alpha: float, step: int, total_steps: int, decay: str) -> float:
    """Compute anchor alpha at a given step — port from DriftEngine.computeAlpha()."""
    if total_steps <= 1:
        return base_alpha
    progress = step / (total_steps - 1)

    if decay == "none":
        return base_alpha
    elif decay == "linear":
        return base_alpha * (1.0 - progress)
    elif decay == "exp":
        return base_alpha * math.exp(-3.0 * progress)
    elif decay == "step":
        return base_alpha if progress < 0.5 else base_alpha * 0.2
    return base_alpha


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def find_top_k(corpus: ProvenanceCorpus, query: np.ndarray, k: int,
               exclude_ids: set[int] | None = None) -> list[tuple[int, float]]:
    """Brute-force dot product retrieval. Returns list of (track_id, similarity)."""
    sims = corpus.embeddings @ query
    if exclude_ids:
        for tid in exclude_ids:
            idx = corpus.tid_to_idx.get(tid)
            if idx is not None:
                sims[idx] = -np.inf

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
    corpus: ProvenanceCorpus,
    lambda_: float
) -> tuple[int, float] | None:
    """MMR single selection."""
    if not candidates:
        return None
    if not selected_embs:
        return candidates[0]

    best_id = -1
    best_mmr = -np.inf
    best_score = 0.0

    for (tid, relevance) in candidates:
        idx = corpus.tid_to_idx.get(tid)
        if idx is None:
            continue
        emb = corpus.embeddings[idx]

        max_sim = max(float(np.dot(emb, sel)) for sel in selected_embs)
        mmr_score = lambda_ * relevance - (1.0 - lambda_) * max_sim

        if mmr_score > best_mmr:
            best_mmr = mmr_score
            best_id = tid
            best_score = relevance

    return (best_id, best_score) if best_id >= 0 else None


def temperature_select_one(
    candidates: list[tuple[int, float]], temperature: float
) -> tuple[int, float] | None:
    """Temperature sampling via Gumbel-max trick."""
    if not candidates:
        return None
    if temperature <= 0:
        return candidates[0]

    scores = np.array([s for _, s in candidates])
    gumbel = -np.log(-np.log(np.random.uniform(0, 1, len(scores))))
    perturbed = scores / temperature + gumbel
    idx = int(np.argmax(perturbed))
    return candidates[idx]


# ---------------------------------------------------------------------------
# Provenance computation
# ---------------------------------------------------------------------------

@dataclass
class Influence:
    source_index: int   # -1 = seed, 0..N-1 = index in result list
    weight: float


@dataclass
class StepResult:
    index: int
    track_id: int
    artist: str | None
    title: str | None
    similarity: float
    influences: list[Influence]
    seed_pct: float
    parent_pct: float


def compute_provenance_interp(
    result_index: int, seed_weight: float
) -> list[Influence]:
    """Compute provenance for seed interpolation.

    Args:
        result_index: 0-based position in results list
        seed_weight: Exact seed weight from the query that found this track
                     (1.0 for the first track, alpha from previous drift step otherwise)
    """
    if result_index == 0:
        return [Influence(-1, 1.0)]

    return [
        Influence(-1, seed_weight),
        Influence(result_index - 1, 1.0 - seed_weight),
    ]


def compute_provenance_ema(
    result_index: int, beta: float, threshold: float = 0.01
) -> list[Influence]:
    """Compute provenance for EMA momentum at a given step."""
    if result_index == 0:
        return [Influence(-1, 1.0)]

    influences = []
    # Seed contribution: beta^result_index
    seed_w = beta ** result_index
    if seed_w > threshold:
        influences.append(Influence(-1, seed_w))

    # Track j contributes beta^(result_index - j - 1) * (1 - beta)
    for j in range(result_index):
        w = (beta ** (result_index - j - 1)) * (1.0 - beta)
        if w > threshold:
            influences.append(Influence(j, w))

    return influences


# ---------------------------------------------------------------------------
# Main provenance simulation
# ---------------------------------------------------------------------------

def run_provenance(
    corpus: ProvenanceCorpus,
    seed_tid: int,
    mode: str = "mmr",
    drift: str = "interp",
    alpha: float = 0.4,
    beta: float = 0.7,
    decay: str = "none",
    temperature: float = 0.05,
    lambda_: float = 0.4,
    num_tracks: int = 20,
    pool_size: int = 50,
    on_progress=None,
) -> list[StepResult]:
    """Run a drift simulation and compute provenance for each step.

    Returns list of StepResult with full influence data.
    """
    def progress(msg):
        if on_progress:
            on_progress(msg)

    seed_idx = corpus.tid_to_idx.get(seed_tid)
    if seed_idx is None:
        raise ValueError(f"Seed track {seed_tid} not found in corpus")

    seed_emb = corpus.embeddings[seed_idx].astype(np.float64)
    query = seed_emb.copy()
    ema_state = None
    seen = {seed_tid}
    results: list[StepResult] = []
    # Track the seed weight of the current query for provenance.
    # Initial query = pure seed, so current_seed_weight = 1.0
    current_seed_weight = 1.0

    for step in range(num_tracks):
        progress(f"Step {step + 1}/{num_tracks}...")

        # Retrieve candidates
        candidates = find_top_k(corpus, query.astype(np.float32), pool_size,
                                exclude_ids=seen)
        if not candidates:
            break

        # Select one
        selected_embs = []
        for r in results:
            idx = corpus.tid_to_idx.get(r.track_id)
            if idx is not None:
                selected_embs.append(corpus.embeddings[idx])

        if mode == "mmr":
            pick = mmr_select_one(candidates, selected_embs, corpus, lambda_)
        elif mode == "temperature":
            pick = temperature_select_one(candidates, temperature)
        else:
            pick = candidates[0] if candidates else None

        if pick is None:
            break

        tid, score = pick
        seen.add(tid)

        # Compute provenance using the seed weight that produced the current query
        result_index = len(results)
        if drift == "interp":
            influences = compute_provenance_interp(result_index, current_seed_weight)
        else:  # ema
            influences = compute_provenance_ema(result_index, beta)

        seed_pct = sum(inf.weight for inf in influences if inf.source_index == -1)
        parent_pct = sum(inf.weight for inf in influences if inf.source_index == result_index - 1) if result_index > 0 else 0.0

        results.append(StepResult(
            index=result_index,
            track_id=tid,
            artist=corpus.artists.get(tid),
            title=corpus.titles.get(tid),
            similarity=score,
            influences=influences,
            seed_pct=seed_pct,
            parent_pct=parent_pct,
        ))

        # Update query for next step and track new seed weight
        current_emb = corpus.embeddings[corpus.tid_to_idx[tid]].astype(np.float64)

        if drift == "interp":
            step_alpha = compute_alpha(alpha, step, num_tracks, decay)
            query = l2_normalize(step_alpha * seed_emb + (1.0 - step_alpha) * current_emb)
            ema_state = current_emb
            current_seed_weight = step_alpha
        else:  # ema
            prev = ema_state if ema_state is not None else seed_emb
            query = l2_normalize(beta * prev + (1.0 - beta) * current_emb)
            ema_state = query.copy()
            current_seed_weight = beta ** (result_index + 1)

    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_provenance(results: list[StepResult]) -> list[str]:
    """Validate invariants on computed provenance. Returns list of error messages."""
    errors = []
    for r in results:
        total = sum(inf.weight for inf in r.influences)
        if abs(total - 1.0) > 0.02:
            errors.append(f"Step {r.index}: influence sum = {total:.4f} (expected ~1.0)")

        for inf in r.influences:
            if inf.weight < 0 or inf.weight > 1.0 + 1e-6:
                errors.append(f"Step {r.index}: influence weight {inf.weight:.4f} out of [0,1]")

            if inf.source_index >= 0 and inf.source_index >= r.index:
                errors.append(f"Step {r.index}: influence from future track {inf.source_index}")

    return errors


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_table(results: list[StepResult], seed_artist: str | None, seed_title: str | None) -> str:
    """Format results as a readable table."""
    lines = []
    lines.append(f"Seed: {seed_artist or '?'} - {seed_title or '?'}")
    lines.append("")
    lines.append(f"{'Step':>4}  {'Track':<40}  {'Sim':>5}  {'Seed%':>6}  {'Parent%':>7}  Influences")
    lines.append("-" * 100)

    for r in results:
        name = f"{r.artist or '?'} - {r.title or '?'}"
        if len(name) > 38:
            name = name[:35] + "..."

        inf_parts = []
        for inf in r.influences:
            label = "seed" if inf.source_index == -1 else f"T{inf.source_index}"
            inf_parts.append(f"{label}:{inf.weight:.3f}")

        lines.append(
            f"{r.index:4d}  {name:<40}  {r.similarity:.3f}  "
            f"{r.seed_pct * 100:5.1f}%  {r.parent_pct * 100:6.1f}%  "
            f"{'  '.join(inf_parts)}"
        )

    return "\n".join(lines)


def format_rail_diagram(results: list[StepResult]) -> str:
    """Format an ASCII rail diagram showing influence structure."""
    if not results:
        return "(empty)"

    # Unicode block characters for opacity levels
    blocks = " \u2591\u2592\u2593\u2588"  # empty, light, medium, dark, full

    def alpha_to_block(alpha: float) -> str:
        if alpha < 0.05:
            return " "
        elif alpha < 0.25:
            return blocks[1]
        elif alpha < 0.50:
            return blocks[2]
        elif alpha < 0.75:
            return blocks[3]
        else:
            return blocks[4]

    lines = []
    max_cols = min(max(r.index for r in results) + 2, 40)  # cap width

    for r in results:
        # Build influence map for this row
        col_alphas: dict[int, float] = {}
        for inf in r.influences:
            col = 0 if inf.source_index == -1 else inf.source_index + 1
            col_alphas[col] = inf.weight

        # Render columns
        cols = []
        for c in range(min(r.index + 2, max_cols)):
            alpha = col_alphas.get(c, 0.0)
            cols.append(alpha_to_block(alpha))

        rail_str = "".join(cols).rstrip()
        name = f"{r.artist or '?'} - {r.title or '?'}"
        if len(name) > 35:
            name = name[:32] + "..."
        lines.append(f"  {rail_str:<{max_cols}}  T{r.index:<3} {name}")

    return "\n".join(lines)
