"""Deep audit: exhaustive multi-model analysis of recommendation quality.

Answers three questions:
1. Does fusing MuLan + Flamingo improve *recommendations* (not just retrieval)?
2. Is temperature sampling salvageable? (score clustering makes Gumbel dominate)
3. What does each embedding model contribute to the final queue?

Usage:
    poweramp-indexer audit embeddings.db --deep [--seeds N] [--quick]

Requires audit_raw_data/ with mulan, flamingo, and fused databases.
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
from scipy import stats as scipy_stats

from .database import EmbeddingDatabase
from .audit import (
    AuditCorpus,
    load_corpus,
    select_seeds,
    find_top_k,
    mmr_select_batch,
    dpp_select_batch,
    temperature_select_batch,
    personalized_pagerank,
    post_filter,
    compute_metrics,
    run_batch_pipeline,
    ValidationResult,
    RunMetrics,
)


# ---------------------------------------------------------------------------
# Multi-model corpus loading
# ---------------------------------------------------------------------------

@dataclass
class MultiModelCorpus:
    """Embeddings from all three models aligned by track_id."""
    # Aligned arrays: same row = same track
    track_ids: np.ndarray       # (N,) int64 — intersection of all models
    mulan: np.ndarray           # (N, 512)
    flamingo: np.ndarray        # (N, 512) — already reduced
    fused: np.ndarray           # (N, 512) — SVD-fused

    artists: dict[int, str | None]
    titles: dict[int, str | None]
    tid_to_idx: dict[int, int]

    # Full fused corpus (for graph, clusters etc.)
    fused_corpus: AuditCorpus


def load_multi_model(
    fused_db_path: Path,
    mulan_db_path: Path,
    flamingo_db_path: Path,
    on_progress=None,
) -> MultiModelCorpus:
    """Load aligned embeddings from all three models."""

    def progress(msg):
        if on_progress:
            on_progress(msg)

    # Load fused corpus (has graph, clusters, metadata)
    progress("Loading fused corpus...")
    fused_db = EmbeddingDatabase(fused_db_path)
    fused_corpus = load_corpus(fused_db, on_progress=on_progress)
    fused_db.close()

    # Load MuLan embeddings
    progress("Loading MuLan embeddings...")
    mulan_db = EmbeddingDatabase(mulan_db_path)

    # Detect table name: legacy DBs use 'embeddings', newer use 'embeddings_mulan'
    mulan_table = "embeddings_mulan"
    tables = [r[0] for r in mulan_db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'embeddings%'"
    ).fetchall()]
    if "embeddings_mulan" in tables:
        mulan_table = "embeddings_mulan"
    elif "embeddings" in tables:
        mulan_table = "embeddings"
    progress(f"  Using table: {mulan_table}")

    mulan_rows = mulan_db.conn.execute(
        f"SELECT track_id, embedding FROM [{mulan_table}]"
    ).fetchall()
    mulan_by_tid = {}
    for row in mulan_rows:
        mulan_by_tid[row["track_id"]] = np.frombuffer(row["embedding"], dtype=np.float32)
    mulan_db.close()
    progress(f"  MuLan: {len(mulan_by_tid)} tracks")

    # Load Flamingo embeddings (reduced to 512d)
    progress("Loading Flamingo embeddings...")
    flam_db = EmbeddingDatabase(flamingo_db_path)

    # The merged DB has both models; get flamingo specifically
    flam_models = flam_db.get_available_models()
    # Prefer 'flamingo', fall back to first non-mulan model
    flam_model = "flamingo"
    if "flamingo" not in flam_models:
        for m in flam_models:
            if m != "mulan":
                flam_model = m
                break

    flam_rows = flam_db.conn.execute(
        f"SELECT track_id, embedding FROM embeddings_{flam_model}"
    ).fetchall()
    flam_by_tid = {}
    for row in flam_rows:
        emb = np.frombuffer(row["embedding"], dtype=np.float32)
        flam_by_tid[row["track_id"]] = emb
    flam_db.close()
    progress(f"  Flamingo ({flam_model}): {len(flam_by_tid)} tracks")

    # Find intersection of all three models
    fused_tids = set(fused_corpus.tid_to_idx.keys())
    mulan_tids = set(mulan_by_tid.keys())
    flam_tids = set(flam_by_tid.keys())
    common_tids = sorted(fused_tids & mulan_tids & flam_tids)
    progress(f"  Intersection: {len(common_tids)} tracks "
             f"(fused={len(fused_tids)}, mulan={len(mulan_tids)}, flamingo={len(flam_tids)})")

    n = len(common_tids)
    dim_mulan = next(iter(mulan_by_tid.values())).shape[0]
    dim_flam = next(iter(flam_by_tid.values())).shape[0]
    dim_fused = fused_corpus.embeddings.shape[1]

    track_ids = np.array(common_tids, dtype=np.int64)
    mulan_embs = np.empty((n, dim_mulan), dtype=np.float32)
    flam_embs = np.empty((n, dim_flam), dtype=np.float32)
    fused_embs = np.empty((n, dim_fused), dtype=np.float32)

    tid_to_idx = {}
    for i, tid in enumerate(common_tids):
        tid_to_idx[tid] = i
        mulan_embs[i] = mulan_by_tid[tid]
        flam_embs[i] = flam_by_tid[tid]
        fused_idx = fused_corpus.tid_to_idx[tid]
        fused_embs[i] = fused_corpus.embeddings[fused_idx]

    return MultiModelCorpus(
        track_ids=track_ids,
        mulan=mulan_embs,
        flamingo=flam_embs,
        fused=fused_embs,
        artists=fused_corpus.artists,
        titles=fused_corpus.titles,
        tid_to_idx=tid_to_idx,
        fused_corpus=fused_corpus,
    )


def _make_corpus_for_model(
    mc: MultiModelCorpus, model: str
) -> AuditCorpus:
    """Create an AuditCorpus using a specific model's embeddings."""
    if model == "fused":
        embs = mc.fused
    elif model == "mulan":
        embs = mc.mulan
    elif model == "flamingo":
        embs = mc.flamingo
    else:
        raise ValueError(f"Unknown model: {model}")

    return AuditCorpus(
        embeddings=embs,
        track_ids=mc.track_ids,
        artists=mc.artists,
        titles=mc.titles,
        tid_to_idx=mc.tid_to_idx,
        idx_to_tid={i: int(mc.track_ids[i]) for i in range(len(mc.track_ids))},
        cluster_centroids=mc.fused_corpus.cluster_centroids,
        cluster_labels=mc.fused_corpus.cluster_labels,
        # Graph only available for fused
        graph_n=mc.fused_corpus.graph_n if model == "fused" else 0,
        graph_k=mc.fused_corpus.graph_k if model == "fused" else 0,
        graph_ids=mc.fused_corpus.graph_ids if model == "fused" else None,
        graph_neighbors=mc.fused_corpus.graph_neighbors if model == "fused" else None,
        graph_weights=mc.fused_corpus.graph_weights if model == "fused" else None,
        graph_tid_to_idx=mc.fused_corpus.graph_tid_to_idx if model == "fused" else {},
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _cluster_spread(tids: list[int], corpus: AuditCorpus) -> int:
    """Count unique clusters in a result set."""
    if corpus.cluster_labels is None:
        return 0
    clusters = set()
    for tid in tids:
        idx = corpus.tid_to_idx.get(tid)
        if idx is not None:
            label = int(corpus.cluster_labels[idx])
            if label >= 0:
                clusters.add(label)
    return len(clusters)


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _ci95(data):
    """95% confidence interval half-width."""
    if len(data) < 2:
        return 0.0
    return 1.96 * np.std(data, ddof=1) / np.sqrt(len(data))


def _wilcoxon_test(a, b):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    diffs = np.array(a) - np.array(b)
    diffs = diffs[diffs != 0]
    if len(diffs) < 10:
        return 0, 1.0
    try:
        stat, p = scipy_stats.wilcoxon(diffs)
        return float(stat), float(p)
    except Exception:
        return 0, 1.0


# ---------------------------------------------------------------------------
# Analysis A: Multi-Model Recommendation Quality Comparison
# ---------------------------------------------------------------------------

def analysis_a(mc: MultiModelCorpus, seeds: list[int], progress) -> dict:
    """Compare recommendation quality across mulan, flamingo, and fused embeddings."""
    progress("\n" + "=" * 70)
    progress("ANALYSIS A: Multi-Model Recommendation Quality Comparison")
    progress("=" * 70)

    models = ["mulan", "flamingo", "fused"]
    corpora = {m: _make_corpus_for_model(mc, m) for m in models}

    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    pool_size = 200
    num_tracks = 50
    max_pa, spacing = 3, 3

    results = {}  # model -> algo_key -> list[RunMetrics]
    cross_overlap = defaultdict(list)  # (algo_key, model_pair) -> [jaccard]
    flamingo_unique_counts = []  # per seed: tracks in fused top-50 not in mulan top-200

    for model in models:
        results[model] = defaultdict(list)

    for si, seed_tid in enumerate(seeds):
        if si % 50 == 0:
            progress(f"  Seed {si}/{len(seeds)}...")

        for model in models:
            corpus = corpora[model]
            seed_idx = corpus.tid_to_idx.get(seed_tid)
            if seed_idx is None:
                continue

            # MMR at multiple lambdas
            for lam in lambdas:
                key = f"mmr_l{lam}"
                res = run_batch_pipeline(
                    seed_tid, corpus, mode="mmr", lambda_=lam,
                    pool_size=pool_size, num_tracks=num_tracks,
                    max_per_artist=max_pa, min_spacing=spacing
                )
                m = compute_metrics(seed_tid, res, corpus, mode="mmr",
                                    config={"lambda": lam})
                m.config["cluster_spread"] = _cluster_spread(m.selected_tids, mc.fused_corpus)
                results[model][key].append(m)

            # DPP
            res = run_batch_pipeline(
                seed_tid, corpus, mode="dpp",
                pool_size=pool_size, num_tracks=num_tracks,
                max_per_artist=max_pa, min_spacing=spacing
            )
            m = compute_metrics(seed_tid, res, corpus, mode="dpp")
            m.config["cluster_spread"] = _cluster_spread(m.selected_tids, mc.fused_corpus)
            results[model]["dpp"].append(m)

        # Cross-model set overlap (fused vs mulan, fused vs flamingo)
        for algo_key in ["mmr_l0.4", "dpp"]:
            fused_tids = set(results["fused"][algo_key][-1].selected_tids) if results["fused"][algo_key] else set()
            mulan_tids = set(results["mulan"][algo_key][-1].selected_tids) if results["mulan"][algo_key] else set()
            flam_tids = set(results["flamingo"][algo_key][-1].selected_tids) if results["flamingo"][algo_key] else set()

            cross_overlap[(algo_key, "fused_vs_mulan")].append(_jaccard(fused_tids, mulan_tids))
            cross_overlap[(algo_key, "fused_vs_flamingo")].append(_jaccard(fused_tids, flam_tids))

        # Flamingo contribution: tracks in fused top-50 NOT in mulan top-200
        fused_corpus = corpora["fused"]
        mulan_corpus = corpora["mulan"]
        seed_idx_fused = fused_corpus.tid_to_idx.get(seed_tid)
        seed_idx_mulan = mulan_corpus.tid_to_idx.get(seed_tid)
        if seed_idx_fused is not None and seed_idx_mulan is not None:
            fused_top50 = find_top_k(fused_corpus, fused_corpus.embeddings[seed_idx_fused], 50,
                                     exclude_ids={seed_tid})
            mulan_top200 = find_top_k(mulan_corpus, mulan_corpus.embeddings[seed_idx_mulan], 200,
                                      exclude_ids={seed_tid})
            mulan_tid_set = {t[0] for t in mulan_top200}
            unique_from_flamingo = sum(1 for t, _ in fused_top50 if t not in mulan_tid_set)
            flamingo_unique_counts.append(unique_from_flamingo)

    # Report
    progress("\n  === MMR λ Sweep (mean across seeds) ===")
    progress(f"  {'Model':<12} {'Lambda':>7} {'sim_seed':>10} {'pairwise':>10} {'artists':>8} {'clusters':>8}")
    for model in models:
        for lam in lambdas:
            key = f"mmr_l{lam}"
            ms = results[model][key]
            if not ms:
                continue
            progress(f"  {model:<12} {lam:>7.1f} "
                     f"{np.mean([m.mean_sim_to_seed for m in ms]):>10.4f} "
                     f"{np.mean([m.mean_pairwise_sim for m in ms]):>10.4f} "
                     f"{np.mean([m.unique_artists for m in ms]):>8.1f} "
                     f"{np.mean([m.config.get('cluster_spread', 0) for m in ms]):>8.1f}")

    progress("\n  === DPP (mean across seeds) ===")
    for model in models:
        ms = results[model]["dpp"]
        if not ms:
            continue
        progress(f"  {model:<12}     "
                 f"sim={np.mean([m.mean_sim_to_seed for m in ms]):.4f} "
                 f"pw={np.mean([m.mean_pairwise_sim for m in ms]):.4f} "
                 f"artists={np.mean([m.unique_artists for m in ms]):.1f} "
                 f"clusters={np.mean([m.config.get('cluster_spread', 0) for m in ms]):.1f}")

    progress("\n  === Cross-Model Set Overlap (Jaccard) ===")
    for (algo, pair), jaccards in sorted(cross_overlap.items()):
        progress(f"  {algo} {pair}: {np.mean(jaccards):.3f} ± {_ci95(jaccards):.3f}")

    progress(f"\n  === Flamingo Contribution ===")
    if flamingo_unique_counts:
        progress(f"  Tracks in fused top-50 NOT in mulan top-200: "
                 f"mean={np.mean(flamingo_unique_counts):.1f}, "
                 f"median={np.median(flamingo_unique_counts):.0f}, "
                 f"max={max(flamingo_unique_counts)}")

    # Statistical tests: fused vs mulan
    progress("\n  === Wilcoxon Signed-Rank: Fused vs MuLan ===")
    for algo_key in ["mmr_l0.4", "dpp"]:
        fused_artists = [m.unique_artists for m in results["fused"][algo_key]]
        mulan_artists = [m.unique_artists for m in results["mulan"][algo_key]]
        fused_pw = [m.mean_pairwise_sim for m in results["fused"][algo_key]]
        mulan_pw = [m.mean_pairwise_sim for m in results["mulan"][algo_key]]

        if len(fused_artists) == len(mulan_artists) and len(fused_artists) > 0:
            _, p_art = _wilcoxon_test(fused_artists, mulan_artists)
            _, p_pw = _wilcoxon_test(fused_pw, mulan_pw)
            progress(f"  {algo_key} unique_artists: fused={np.mean(fused_artists):.1f} vs mulan={np.mean(mulan_artists):.1f}, p={p_art:.4f}")
            progress(f"  {algo_key} pairwise_sim:   fused={np.mean(fused_pw):.4f} vs mulan={np.mean(mulan_pw):.4f}, p={p_pw:.4f}")

    return {
        "results": {model: {k: len(v) for k, v in d.items()} for model, d in results.items()},
        "cross_overlap": {f"{a}|{p}": float(np.mean(j)) for (a, p), j in cross_overlap.items()},
        "flamingo_unique_mean": float(np.mean(flamingo_unique_counts)) if flamingo_unique_counts else 0,
        "flamingo_unique_median": float(np.median(flamingo_unique_counts)) if flamingo_unique_counts else 0,
    }


# ---------------------------------------------------------------------------
# Analysis B: Temperature Score Distribution
# ---------------------------------------------------------------------------

def analysis_b(mc: MultiModelCorpus, seeds: list[int], progress) -> dict:
    """Exhaustive temperature analysis with transformation comparison."""
    progress("\n" + "=" * 70)
    progress("ANALYSIS B: Temperature Score Distribution")
    progress("=" * 70)

    corpus = _make_corpus_for_model(mc, "fused")
    pool_sizes = [200, 500, 1000]

    # Part 1: Score distribution by pool size
    progress("\n  --- Score Distribution by Pool Size ---")
    score_stats = {}
    all_scores_by_pool = defaultdict(list)

    for si, seed_tid in enumerate(seeds):
        seed_idx = corpus.tid_to_idx.get(seed_tid)
        if seed_idx is None:
            continue
        seed_emb = corpus.embeddings[seed_idx]

        for pool in pool_sizes:
            candidates = find_top_k(corpus, seed_emb, pool, exclude_ids={seed_tid})
            scores = [s for _, s in candidates]
            all_scores_by_pool[pool].extend(scores)

            if si == 0:  # Show first seed as example
                score_stats[pool] = {
                    "range": float(max(scores) - min(scores)),
                    "iqr": float(np.percentile(scores, 75) - np.percentile(scores, 25)),
                    "std": float(np.std(scores)),
                    "min": float(min(scores)),
                    "max": float(max(scores)),
                }

    for pool in pool_sizes:
        scores = all_scores_by_pool[pool]
        arr = np.array(scores)
        progress(f"  Pool {pool}: range={np.ptp(arr):.4f}, "
                 f"IQR={np.percentile(arr, 75) - np.percentile(arr, 25):.4f}, "
                 f"std={np.std(arr):.4f}, "
                 f"skew={scipy_stats.skew(arr):.2f}")

    # Part 2: Transformation comparison
    progress("\n  --- Transformation Comparison ---")

    taus = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    n_runs = 20
    pool = 200
    num_tracks = 50
    max_pa, spacing = 3, 3

    def _raw_transform(scores, T):
        return scores / T

    def _rank_transform(scores, T):
        n = len(scores)
        ranks = np.argsort(np.argsort(-scores)).astype(np.float64)
        return (1.0 - ranks / n) / T

    def _minmax_transform(scores, T):
        mn, mx = scores.min(), scores.max()
        if mx - mn < 1e-10:
            return scores / T
        return ((scores - mn) / (mx - mn)) / T

    def _log_transform(scores, T):
        # scores are cosine sims, typically 0.8-1.0
        return np.log(np.clip(scores, 1e-10, None)) / T

    def _percentile_transform(scores, T):
        n = len(scores)
        ranks = scipy_stats.rankdata(scores) / n
        return ranks / T

    def _power_transform(scores, T, p=4):
        return (scores ** p) / T

    transforms = {
        "raw": _raw_transform,
        "rank": _rank_transform,
        "minmax": _minmax_transform,
        "log": _log_transform,
        "percentile": _percentile_transform,
        "power4": lambda s, T: _power_transform(s, T, p=4),
        "power8": lambda s, T: _power_transform(s, T, p=8),
    }

    # For each transform x T, run n_runs selections and measure overlap
    transform_results = {}  # transform_name -> tau -> {jaccard_mean, jaccard_std, sim_mean, artists_mean}

    for tname, tfunc in transforms.items():
        transform_results[tname] = {}
        for tau in taus:
            all_jaccards = []
            all_sims = []
            all_artists = []

            for si, seed_tid in enumerate(seeds[:100]):  # Use 100 seeds for speed
                seed_idx = corpus.tid_to_idx.get(seed_tid)
                if seed_idx is None:
                    continue
                seed_emb = corpus.embeddings[seed_idx]
                candidates = find_top_k(corpus, seed_emb, pool, exclude_ids={seed_tid})
                if len(candidates) < 10:
                    continue

                # Transform scores
                cand_ids = np.array([c[0] for c in candidates])
                cand_scores = np.array([c[1] for c in candidates], dtype=np.float64)
                transformed = tfunc(cand_scores, tau)

                # Run n_runs with Gumbel noise
                run_selections = []
                for run in range(n_runs):
                    rng = np.random.default_rng(42 + si * 10000 + run)
                    gumbels = -np.log(-np.log(np.clip(rng.random(len(transformed)), 1e-10, 1 - 1e-10)))
                    perturbed = transformed + gumbels

                    # Select top-num_tracks without replacement (iterative for post-filter)
                    selected_tids = []
                    selected_sims = []
                    remaining_indices = list(range(len(perturbed)))

                    for _ in range(min(num_tracks, len(remaining_indices))):
                        if not remaining_indices:
                            break
                        best_ri = max(remaining_indices, key=lambda ri: perturbed[ri])
                        remaining_indices.remove(best_ri)
                        selected_tids.append(int(cand_ids[best_ri]))
                        selected_sims.append(float(cand_scores[best_ri]))

                    # Post-filter
                    filtered = post_filter(
                        list(zip(selected_tids, selected_sims)),
                        corpus, max_pa, spacing
                    )
                    run_selections.append(set(t for t, _ in filtered))
                    all_sims.append(np.mean([s for _, s in filtered]) if filtered else 0)
                    known = [corpus.artists.get(t) for t, _ in filtered if corpus.artists.get(t)]
                    all_artists.append(len(set(known)))

                # Pairwise Jaccard across runs
                for a in range(len(run_selections)):
                    for b in range(a + 1, len(run_selections)):
                        all_jaccards.append(_jaccard(run_selections[a], run_selections[b]))

            transform_results[tname][tau] = {
                "jaccard_mean": float(np.mean(all_jaccards)) if all_jaccards else 0,
                "jaccard_std": float(np.std(all_jaccards)) if all_jaccards else 0,
                "sim_mean": float(np.mean(all_sims)) if all_sims else 0,
                "artists_mean": float(np.mean(all_artists)) if all_artists else 0,
            }

        # Summary per transform
        j_at_min_t = transform_results[tname][taus[0]]["jaccard_mean"]
        j_at_max_t = transform_results[tname][taus[-1]]["jaccard_mean"]
        dynamic_range = j_at_min_t / max(j_at_max_t, 1e-10)

        # Check monotonicity of Jaccard vs T
        jaccards_by_t = [transform_results[tname][t]["jaccard_mean"] for t in taus]
        monotonic_violations = sum(1 for i in range(1, len(jaccards_by_t))
                                   if jaccards_by_t[i] > jaccards_by_t[i-1] + 0.01)

        progress(f"  {tname:<12}: J(T_min)={j_at_min_t:.3f}, J(T_max)={j_at_max_t:.3f}, "
                 f"dynamic_range={dynamic_range:.1f}x, mono_violations={monotonic_violations}")

    # Find best transform (highest dynamic range with monotonicity)
    best_name = None
    best_range = 0
    for tname, tdata in transform_results.items():
        jaccards_by_t = [tdata[t]["jaccard_mean"] for t in taus]
        mono_violations = sum(1 for i in range(1, len(jaccards_by_t))
                              if jaccards_by_t[i] > jaccards_by_t[i-1] + 0.01)
        j_min = tdata[taus[0]]["jaccard_mean"]
        j_max = tdata[taus[-1]]["jaccard_mean"]
        drange = j_min / max(j_max, 1e-10)
        if mono_violations <= 1 and drange > best_range:
            best_range = drange
            best_name = tname

    progress(f"\n  >>> Best transform: {best_name} (dynamic range {best_range:.1f}x)")

    # Print controllability curve for best
    if best_name:
        progress(f"\n  Controllability curve ({best_name}):")
        progress(f"  {'T':>8} {'Jaccard':>10} {'Sim':>10} {'Artists':>10}")
        for tau in taus:
            d = transform_results[best_name][tau]
            progress(f"  {tau:>8.3f} {d['jaccard_mean']:>10.3f} {d['sim_mean']:>10.4f} {d['artists_mean']:>10.1f}")

    return {
        "score_distribution": {p: {k: float(v) for k, v in s.items()} for p, s in score_stats.items()},
        "transform_results": transform_results,
        "best_transform": best_name,
        "best_dynamic_range": best_range,
    }


# ---------------------------------------------------------------------------
# Analysis C: Per-Model Embedding Space Profiling
# ---------------------------------------------------------------------------

def analysis_c(mc: MultiModelCorpus, progress) -> dict:
    """Profile each embedding space: pairwise similarity, discriminability, neighbor agreement."""
    progress("\n" + "=" * 70)
    progress("ANALYSIS C: Per-Model Embedding Space Profiling")
    progress("=" * 70)

    n = len(mc.track_ids)
    rng = np.random.default_rng(42)
    models = {"mulan": mc.mulan, "flamingo": mc.flamingo, "fused": mc.fused}

    # Part 1: Global pairwise similarity distribution
    progress("\n  --- Global Pairwise Similarity (5000 random pairs) ---")
    n_pairs = 5000
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    pairwise_stats = {}
    for mname, embs in models.items():
        sims = np.sum(embs[idx_a] * embs[idx_b], axis=1)
        percentiles = np.percentile(sims, [1, 5, 25, 50, 75, 95, 99])
        pairwise_stats[mname] = {
            "mean": float(np.mean(sims)),
            "std": float(np.std(sims)),
            "p1": float(percentiles[0]),
            "p5": float(percentiles[1]),
            "p25": float(percentiles[2]),
            "p50": float(percentiles[3]),
            "p75": float(percentiles[4]),
            "p95": float(percentiles[5]),
            "p99": float(percentiles[6]),
        }
        progress(f"  {mname:<12}: mean={pairwise_stats[mname]['mean']:.4f}, "
                 f"std={pairwise_stats[mname]['std']:.4f}, "
                 f"p5={pairwise_stats[mname]['p5']:.4f}, "
                 f"p50={pairwise_stats[mname]['p50']:.4f}, "
                 f"p95={pairwise_stats[mname]['p95']:.4f}")

    # Part 2: Same-artist vs different-artist
    progress("\n  --- Same-Artist vs Different-Artist Similarity ---")

    # Build artist groups
    artist_to_indices = defaultdict(list)
    for i, tid in enumerate(mc.track_ids):
        artist = mc.artists.get(int(tid))
        if artist:
            artist_to_indices[artist].append(i)

    # Sample same-artist pairs
    same_pairs_a, same_pairs_b = [], []
    multi_artists = [a for a, idxs in artist_to_indices.items() if len(idxs) >= 2]
    for _ in range(5000):
        artist = rng.choice(multi_artists)
        idxs = artist_to_indices[artist]
        a, b = rng.choice(len(idxs), size=2, replace=False)
        same_pairs_a.append(idxs[a])
        same_pairs_b.append(idxs[b])
    same_pairs_a = np.array(same_pairs_a)
    same_pairs_b = np.array(same_pairs_b)

    # Sample different-artist pairs
    diff_pairs_a = rng.integers(0, n, size=5000)
    diff_pairs_b = rng.integers(0, n, size=5000)
    # Filter to ensure different artists
    mask = np.array([
        mc.artists.get(int(mc.track_ids[a])) != mc.artists.get(int(mc.track_ids[b]))
        for a, b in zip(diff_pairs_a, diff_pairs_b)
    ])
    diff_pairs_a = diff_pairs_a[mask][:5000]
    diff_pairs_b = diff_pairs_b[mask][:5000]

    discriminability = {}
    for mname, embs in models.items():
        same_sims = np.sum(embs[same_pairs_a] * embs[same_pairs_b], axis=1)
        diff_sims = np.sum(embs[diff_pairs_a] * embs[diff_pairs_b], axis=1)
        mean_same = float(np.mean(same_sims))
        mean_diff = float(np.mean(diff_sims))
        pooled_std = float(np.sqrt((np.var(same_sims) + np.var(diff_sims)) / 2))
        d_index = (mean_same - mean_diff) / pooled_std if pooled_std > 0 else 0
        discriminability[mname] = {
            "mean_same": mean_same,
            "mean_diff": mean_diff,
            "d_index": d_index,
        }
        progress(f"  {mname:<12}: same_artist={mean_same:.4f}, diff_artist={mean_diff:.4f}, "
                 f"d'={d_index:.3f}")

    # Part 3: Cross-model neighbor agreement
    progress("\n  --- Cross-Model Neighbor Agreement (500 seeds) ---")
    n_check = 500
    k_nn = 20
    check_indices = rng.choice(n, size=min(n_check, n), replace=False)

    overlaps = defaultdict(list)
    for idx in check_indices:
        nns = {}
        for mname, embs in models.items():
            sims = embs @ embs[idx]
            sims[idx] = -np.inf  # exclude self
            top_k = np.argpartition(sims, -k_nn)[-k_nn:]
            nns[mname] = set(top_k.tolist())

        overlaps["mulan_flamingo"].append(len(nns["mulan"] & nns["flamingo"]) / k_nn)
        overlaps["fused_mulan"].append(len(nns["fused"] & nns["mulan"]) / k_nn)
        overlaps["fused_flamingo"].append(len(nns["fused"] & nns["flamingo"]) / k_nn)

    for pair, vals in overlaps.items():
        progress(f"  {pair:<20}: mean overlap = {np.mean(vals):.3f} ± {_ci95(vals):.3f} "
                 f"(out of top-{k_nn})")

    # Part 4: Cluster-level analysis
    progress("\n  --- Cluster-Level Model Agreement ---")
    if mc.fused_corpus.cluster_labels is not None:
        n_clusters = len(mc.fused_corpus.cluster_centroids) if mc.fused_corpus.cluster_centroids is not None else 0
        flamingo_dominant = 0
        mulan_dominant = 0

        for k in range(n_clusters):
            # Find tracks in this cluster (map from fused corpus indices to multi-model indices)
            fused_mask = mc.fused_corpus.cluster_labels == k
            fused_cluster_tids = set(mc.fused_corpus.track_ids[fused_mask].tolist())
            cluster_indices = [mc.tid_to_idx[tid] for tid in fused_cluster_tids if tid in mc.tid_to_idx]

            if len(cluster_indices) < 3:
                continue

            ci = np.array(cluster_indices)
            for mname, embs in models.items():
                if mname == "fused":
                    continue
                cluster_embs = embs[ci]
                pw = cluster_embs @ cluster_embs.T
                n_c = len(ci)
                triu = np.triu_indices(n_c, k=1)
                mean_pw = float(np.mean(pw[triu]))

                if mname == "mulan" and mean_pw > 0.7:
                    mulan_dominant += 1
                elif mname == "flamingo" and mean_pw > 0.95:
                    flamingo_dominant += 1

        progress(f"  Clusters with high MuLan agreement (pw > 0.7): {mulan_dominant}/{n_clusters}")
        progress(f"  Clusters with high Flamingo agreement (pw > 0.95): {flamingo_dominant}/{n_clusters}")

    return {
        "pairwise_stats": pairwise_stats,
        "discriminability": discriminability,
        "neighbor_overlap": {k: float(np.mean(v)) for k, v in overlaps.items()},
    }


# ---------------------------------------------------------------------------
# Analysis D: Algorithm × Embedding Fitness Matrix
# ---------------------------------------------------------------------------

def analysis_d(mc: MultiModelCorpus, seeds: list[int], progress) -> dict:
    """Run all algorithms on all embedding spaces, find best combinations."""
    progress("\n" + "=" * 70)
    progress("ANALYSIS D: Algorithm × Embedding Fitness Matrix")
    progress("=" * 70)

    model_names = ["mulan", "flamingo", "fused"]
    corpora = {m: _make_corpus_for_model(mc, m) for m in model_names}
    algos = {
        "mmr_0.4": ("mmr", {"lambda_": 0.4}),
        "dpp": ("dpp", {}),
        "temp_0.05": ("temperature", {"temperature": 0.05}),
    }
    # Random walk only on fused (has graph)
    pool_size = 200
    num_tracks = 50
    max_pa, spacing = 3, 3

    matrix = {}  # (algo, model) -> list[RunMetrics]

    for algo_name, (mode, kwargs) in algos.items():
        for model in model_names:
            key = (algo_name, model)
            matrix[key] = []
            corpus = corpora[model]

            for si, seed_tid in enumerate(seeds):
                if si % 50 == 0 and si > 0:
                    progress(f"  {algo_name}/{model}: seed {si}/{len(seeds)}...")

                rng = np.random.default_rng(42 + si) if mode == "temperature" else None
                res = run_batch_pipeline(
                    seed_tid, corpus, mode=mode,
                    pool_size=pool_size, num_tracks=num_tracks,
                    max_per_artist=max_pa, min_spacing=spacing,
                    rng=rng, **kwargs
                )
                m = compute_metrics(seed_tid, res, corpus, mode=mode)
                m.config["cluster_spread"] = _cluster_spread(m.selected_tids, mc.fused_corpus)
                matrix[key].append(m)

    # Random Walk on fused only
    fused_corpus = corpora["fused"]
    matrix[("random_walk", "fused")] = []
    for si, seed_tid in enumerate(seeds):
        res = run_batch_pipeline(
            seed_tid, fused_corpus, mode="random_walk", alpha=0.5,
            pool_size=pool_size, num_tracks=num_tracks,
            max_per_artist=max_pa, min_spacing=spacing
        )
        m = compute_metrics(seed_tid, res, fused_corpus, mode="random_walk")
        m.config["cluster_spread"] = _cluster_spread(m.selected_tids, mc.fused_corpus)
        matrix[("random_walk", "fused")].append(m)

    # Report
    progress(f"\n  {'Algorithm':<15} {'Model':<12} {'sim_seed':>10} {'pairwise':>10} {'artists':>8} {'clusters':>8} {'len':>5}")
    progress("  " + "-" * 70)
    for (algo, model), ms in sorted(matrix.items()):
        if not ms:
            continue
        progress(f"  {algo:<15} {model:<12} "
                 f"{np.mean([m.mean_sim_to_seed for m in ms]):>10.4f} "
                 f"{np.mean([m.mean_pairwise_sim for m in ms]):>10.4f} "
                 f"{np.mean([m.unique_artists for m in ms]):>8.1f} "
                 f"{np.mean([m.config.get('cluster_spread', 0) for m in ms]):>8.1f} "
                 f"{np.mean([m.queue_length for m in ms]):>5.1f}")

    # Which algorithm benefits most from fusion?
    progress("\n  === Fusion Benefit (fused - mulan) ===")
    for algo_name in algos:
        fused_ms = matrix.get((algo_name, "fused"), [])
        mulan_ms = matrix.get((algo_name, "mulan"), [])
        if fused_ms and mulan_ms and len(fused_ms) == len(mulan_ms):
            art_diff = np.mean([f.unique_artists for f in fused_ms]) - np.mean([m.unique_artists for m in mulan_ms])
            pw_diff = np.mean([f.mean_pairwise_sim for f in fused_ms]) - np.mean([m.mean_pairwise_sim for m in mulan_ms])
            progress(f"  {algo_name:<15}: artists Δ={art_diff:+.1f}, pairwise Δ={pw_diff:+.4f}")

    # Any seeds where flamingo beats fused?
    progress("\n  === Seeds Where Flamingo-Only Beats Fused ===")
    for algo_name in algos:
        fused_ms = matrix.get((algo_name, "fused"), [])
        flam_ms = matrix.get((algo_name, "flamingo"), [])
        if fused_ms and flam_ms:
            count = sum(1 for f, fl in zip(fused_ms, flam_ms) if fl.unique_artists > f.unique_artists + 2)
            progress(f"  {algo_name}: {count}/{len(fused_ms)} seeds where flamingo has 3+ more artists")

    return {
        "matrix": {
            f"{a}|{m}": {
                "sim_mean": float(np.mean([x.mean_sim_to_seed for x in ms])),
                "pw_mean": float(np.mean([x.mean_pairwise_sim for x in ms])),
                "artists_mean": float(np.mean([x.unique_artists for x in ms])),
                "clusters_mean": float(np.mean([x.config.get("cluster_spread", 0) for x in ms])),
            }
            for (a, m), ms in matrix.items() if ms
        }
    }


# ---------------------------------------------------------------------------
# Analysis E: Knob Sensitivity Deep Sweep
# ---------------------------------------------------------------------------

def analysis_e(mc: MultiModelCorpus, seeds: list[int], progress) -> dict:
    """Fine-grained parameter sweeps on fused embeddings."""
    progress("\n" + "=" * 70)
    progress("ANALYSIS E: Knob Sensitivity Deep Sweep")
    progress("=" * 70)

    corpus = _make_corpus_for_model(mc, "fused")
    pool_size = 200
    num_tracks = 50

    # MMR λ: 21 points
    lambdas = [round(i * 0.05, 2) for i in range(21)]
    filter_configs = [
        ("filtered", 3, 3),
        ("unfiltered", 999, 0),
    ]

    progress("\n  --- MMR λ Sweep (21 points × 2 filter configs) ---")

    lambda_results = {}  # (filter_name, lambda) -> {sim, pw, artists, clusters}

    for fname, max_pa, spacing in filter_configs:
        for lam in lambdas:
            sims, pws, arts, clusters = [], [], [], []
            for si, seed_tid in enumerate(seeds):
                res = run_batch_pipeline(
                    seed_tid, corpus, mode="mmr", lambda_=lam,
                    pool_size=pool_size, num_tracks=num_tracks,
                    max_per_artist=max_pa, min_spacing=spacing
                )
                m = compute_metrics(seed_tid, res, corpus, mode="mmr")
                sims.append(m.mean_sim_to_seed)
                pws.append(m.mean_pairwise_sim)
                arts.append(m.unique_artists)
                clusters.append(_cluster_spread(m.selected_tids, mc.fused_corpus))

            lambda_results[(fname, lam)] = {
                "sim": float(np.mean(sims)),
                "pw": float(np.mean(pws)),
                "artists": float(np.mean(arts)),
                "clusters": float(np.mean(clusters)),
            }

        # Report
        progress(f"\n  {fname}:")
        progress(f"  {'Lambda':>7} {'sim_seed':>10} {'pairwise':>10} {'artists':>8} {'clusters':>8}")
        for lam in lambdas[::4]:  # Print every 4th for readability
            r = lambda_results[(fname, lam)]
            progress(f"  {lam:>7.2f} {r['sim']:>10.4f} {r['pw']:>10.4f} {r['artists']:>8.1f} {r['clusters']:>8.1f}")

    # Pareto frontier
    progress("\n  --- Pareto Frontier (sim vs diversity) ---")
    pareto = []
    for lam in lambdas:
        r = lambda_results[("filtered", lam)]
        pareto.append((lam, r["sim"], 1.0 - r["pw"]))  # diversity = 1 - pairwise_sim
    # Find Pareto-optimal points
    optimal = []
    for i, (lam, sim, div) in enumerate(pareto):
        dominated = False
        for j, (lam2, sim2, div2) in enumerate(pareto):
            if i != j and sim2 >= sim and div2 >= div and (sim2 > sim or div2 > div):
                dominated = True
                break
        if not dominated:
            optimal.append((lam, sim, div))
    progress(f"  Pareto-optimal λ values: {[o[0] for o in optimal]}")

    # Random Walk α sweep: 21 points
    progress("\n  --- Random Walk α Sweep (21 points) ---")
    alphas = [round(i * 0.05, 2) for i in range(1, 21)]  # 0.05 to 1.0

    rw_results = {}
    for alpha in alphas:
        sims, arts, transitives = [], [], []
        for si, seed_tid in enumerate(seeds):
            res = run_batch_pipeline(
                seed_tid, corpus, mode="random_walk", alpha=alpha,
                pool_size=pool_size, num_tracks=num_tracks,
                max_per_artist=3, min_spacing=3
            )
            m = compute_metrics(seed_tid, res, corpus, mode="random_walk")

            # Transitive count
            seed_idx = corpus.tid_to_idx.get(seed_tid)
            if seed_idx is not None:
                bf = find_top_k(corpus, corpus.embeddings[seed_idx], 200, exclude_ids={seed_tid})
                bf_tids = {t for t, _ in bf}
                tc = sum(1 for t, _ in res if t not in bf_tids)
                transitives.append(tc)

            sims.append(m.mean_sim_to_seed)
            arts.append(m.unique_artists)

        rw_results[alpha] = {
            "sim": float(np.mean(sims)),
            "artists": float(np.mean(arts)),
            "transitive": float(np.mean(transitives)) if transitives else 0,
        }

    progress(f"  {'Alpha':>7} {'sim_seed':>10} {'artists':>8} {'transitive':>10}")
    for alpha in alphas[::4]:
        r = rw_results[alpha]
        progress(f"  {alpha:>7.2f} {r['sim']:>10.4f} {r['artists']:>8.1f} {r['transitive']:>10.1f}")

    return {
        "lambda_sweep": {f"{fn}_{lam}": v for (fn, lam), v in lambda_results.items()},
        "pareto_optimal": [o[0] for o in optimal],
        "rw_alpha_sweep": rw_results,
    }


# ---------------------------------------------------------------------------
# Main deep audit runner
# ---------------------------------------------------------------------------

def run_deep_audit(
    fused_db_path: Path,
    raw_data_dir: Path | None = None,
    n_seeds: int = 200,
    quick: bool = False,
    on_progress=None,
):
    """Run the full deep audit.

    Args:
        fused_db_path: Path to the fused embedding database
        raw_data_dir: Directory containing embeddings_mulan.db,
                      embeddings_flamingo.db, embeddings-flam-mulan-full-reduced.db.
                      If None, tries audit_raw_data/ relative to fused_db_path.
        n_seeds: Number of seed tracks
        quick: Reduce seeds and grid resolution
    """

    def progress(msg):
        print(msg, flush=True)
        if on_progress:
            on_progress(msg)

    start_time = time.time()

    if quick:
        n_seeds = min(n_seeds, 50)

    # Find raw data directory
    if raw_data_dir is None:
        raw_data_dir = fused_db_path.parent / "audit_raw_data"

    mulan_db_path = raw_data_dir / "embeddings_mulan.db"
    # The merged DB has flamingo at 512d (already reduced)
    merged_db_path = raw_data_dir / "embeddings-flam-mulan-full-reduced.db"

    if not mulan_db_path.exists():
        raise FileNotFoundError(f"MuLan DB not found: {mulan_db_path}")
    if not merged_db_path.exists():
        raise FileNotFoundError(f"Merged DB not found: {merged_db_path}")

    progress("=" * 70)
    progress("DEEP AUDIT: Multi-Model Recommendation Quality Analysis")
    progress("=" * 70)

    # Load multi-model corpus
    mc = load_multi_model(
        fused_db_path=fused_db_path,
        mulan_db_path=mulan_db_path,
        flamingo_db_path=merged_db_path,
        on_progress=progress,
    )

    progress(f"\nCorpus: {len(mc.track_ids)} aligned tracks, "
             f"mulan={mc.mulan.shape[1]}d, flamingo={mc.flamingo.shape[1]}d, fused={mc.fused.shape[1]}d")

    # Select seeds from fused corpus (uses cluster centroids + random fill)
    seeds = select_seeds(mc.fused_corpus, n_seeds)

    # Add random seeds to reach 300 if n_seeds >= 300
    if n_seeds >= 300 and len(seeds) < 300:
        rng = np.random.default_rng(123)
        all_tids = set(mc.track_ids.tolist())
        existing = set(seeds)
        remaining = list(all_tids - existing)
        rng.shuffle(remaining)
        seeds.extend(remaining[:300 - len(seeds)])

    progress(f"Selected {len(seeds)} seeds\n")

    all_results = {}

    # Analysis A
    all_results["A"] = analysis_a(mc, seeds, progress)

    # Analysis B
    all_results["B"] = analysis_b(mc, seeds[:200], progress)

    # Analysis C
    all_results["C"] = analysis_c(mc, progress)

    # Analysis D
    all_results["D"] = analysis_d(mc, seeds, progress)

    # Analysis E
    all_results["E"] = analysis_e(mc, seeds, progress)

    # Summary
    elapsed = time.time() - start_time
    progress("\n" + "=" * 70)
    progress("DEEP AUDIT COMPLETE")
    progress("=" * 70)
    progress(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")

    # Key findings
    progress("\n--- KEY FINDINGS ---")

    # Q1: Does fusion help?
    a_data = all_results["A"]
    progress(f"\nQ1: Does fusion improve recommendations?")
    overlap = a_data.get("cross_overlap", {})
    if overlap:
        for key, val in sorted(overlap.items()):
            progress(f"  {key}: Jaccard = {val:.3f}")
    flamingo_contrib = a_data.get("flamingo_unique_mean", 0)
    progress(f"  Flamingo-unique tracks in fused top-50: {flamingo_contrib:.1f} per seed")

    # Q2: Is temperature salvageable?
    b_data = all_results["B"]
    progress(f"\nQ2: Is temperature sampling salvageable?")
    best_transform = b_data.get("best_transform", "unknown")
    best_range = b_data.get("best_dynamic_range", 0)
    progress(f"  Best transform: {best_transform} (dynamic range: {best_range:.1f}x)")
    if best_range > 3:
        progress(f"  >> YES — {best_transform} gives {best_range:.1f}x dynamic range. Implement it.")
    else:
        progress(f"  >> MARGINAL — only {best_range:.1f}x. Consider removing temperature mode.")

    # Q3: What does each model contribute?
    c_data = all_results["C"]
    progress(f"\nQ3: What does each model contribute?")
    for mname, disc in c_data.get("discriminability", {}).items():
        progress(f"  {mname}: d'={disc['d_index']:.3f}")
    for pair, overlap in c_data.get("neighbor_overlap", {}).items():
        progress(f"  NN overlap {pair}: {overlap:.3f}")

    # Write JSON results
    json_path = fused_db_path.parent / "deep_audit_results.json"
    progress(f"\nWriting detailed results to {json_path}")

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
        json.dump({
            "elapsed_seconds": elapsed,
            "n_seeds": len(seeds),
            "n_tracks": len(mc.track_ids),
            **all_results,
        }, f, indent=2, cls=NumpyEncoder)

    return all_results
