#!/usr/bin/env python3
"""
Validate phone benchmark embeddings against the desktop DB.

Usage:
    python validate_benchmark.py benchmark_results.json fused.db

Compares MuLan and Flamingo embeddings from the phone's BenchmarkActivity
output against the desktop-generated embeddings stored in the fused DB.
For Flamingo, applies the stored projection matrix (3584d → 512d) to
match the phone's raw output against the desktop's reduced embeddings.

Expected results (based on prior TFLite vs desktop validation):
  MuLan:   ~0.982 cosine similarity
  Flamingo: ~0.990 cosine similarity
"""

import json
import sqlite3
import struct
import sys
from difflib import SequenceMatcher

import numpy as np


def load_benchmark(path: str) -> dict:
    """Load benchmark_results.json from the phone (handles UTF-16/BOM)."""
    raw = open(path, "rb").read()
    # Handle UTF-16 LE/BE BOM
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    # Handle UTF-8 BOM
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8")
    else:
        text = raw.decode("utf-8")
    # Strip any remaining BOM character
    text = text.lstrip("\ufeff")
    return json.loads(text)


def load_db_embeddings(db_path: str, track_ids: list[int], table: str) -> dict[int, np.ndarray]:
    """Load embeddings for specific track IDs from the DB."""
    db = sqlite3.connect(db_path)
    result = {}
    for tid in track_ids:
        row = db.execute(
            f"SELECT embedding FROM {table} WHERE track_id = ?", (tid,)
        ).fetchone()
        if row:
            blob = row[0]
            arr = np.frombuffer(blob, dtype=np.float32).copy()
            result[tid] = arr
    db.close()
    return result


def load_projection_matrix(db_path: str, key: str) -> np.ndarray | None:
    """Load a projection matrix from the metadata table."""
    db = sqlite3.connect(db_path)
    row = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    db.close()
    if row is None:
        return None
    blob = row[0]
    n_floats = len(blob) // 4
    return np.frombuffer(blob, dtype=np.float32).reshape(-1).copy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def match_track_in_db(artist: str, title: str, db_path: str) -> list[tuple[int, str, str, float]]:
    """
    Find the best matching track(s) in the desktop DB by artist+title.
    Returns list of (track_id, db_artist, db_title, match_score).
    """
    db = sqlite3.connect(db_path)
    cur = db.cursor()

    # Normalize for matching
    artist_norm = artist.strip().lower()
    title_norm = title.strip().lower().rstrip(".")

    # Try exact match first
    cur.execute(
        "SELECT id, artist, title FROM tracks WHERE LOWER(artist) = ? AND LOWER(TRIM(title, '.')) = ?",
        (artist_norm, title_norm),
    )
    rows = cur.fetchall()
    if rows:
        db.close()
        return [(r[0], r[1], r[2], 1.0) for r in rows]

    # Fuzzy match by title similarity
    cur.execute(
        "SELECT id, artist, title FROM tracks WHERE LOWER(title) LIKE ?",
        (f"%{title_norm[:20]}%",),
    )
    candidates = cur.fetchall()
    db.close()

    scored = []
    for tid, db_artist, db_title in candidates:
        # Score by combined artist+title similarity
        artist_score = SequenceMatcher(None, artist_norm, db_artist.lower()).ratio()
        title_score = SequenceMatcher(None, title_norm, db_title.lower().rstrip(".")).ratio()
        combined = 0.4 * artist_score + 0.6 * title_score
        if combined > 0.5:
            scored.append((tid, db_artist, db_title, combined))

    scored.sort(key=lambda x: -x[3])
    return scored[:3]


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} benchmark_results.json fused.db")
        sys.exit(1)

    benchmark_path = sys.argv[1]
    db_path = sys.argv[2]

    # Load benchmark results
    benchmark = load_benchmark(benchmark_path)
    print(f"Benchmark: {benchmark['device']} ({benchmark['soc']})")
    print(f"Runtime: {benchmark['runtime']}")
    print(f"MuLan EP: {benchmark.get('mulanEp', 'N/A')}")
    print(f"Flamingo EP: {benchmark.get('flamingoEp', 'N/A')}")
    print(f"Tracks: {len(benchmark['tracks'])}")
    print()

    # Load projection matrices for Flamingo comparison
    db = sqlite3.connect(db_path)
    flamingo_proj_blob = db.execute(
        "SELECT value FROM metadata WHERE key = 'flamingo_projection'"
    ).fetchone()
    fused_proj_blob = db.execute(
        "SELECT value FROM metadata WHERE key = 'fused_projection'"
    ).fetchone()
    flamingo_orig_dim = int(
        db.execute("SELECT value FROM metadata WHERE key = 'flamingo_original_dim'").fetchone()[0]
    )
    flamingo_red_dim = int(
        db.execute("SELECT value FROM metadata WHERE key = 'flamingo_reduced_dim'").fetchone()[0]
    )
    db.close()

    # flamingo_projection: V_k[3584, 512] stored row-major
    flamingo_proj = None
    if flamingo_proj_blob:
        flamingo_proj = np.frombuffer(flamingo_proj_blob[0], dtype=np.float32).copy()
        flamingo_proj = flamingo_proj.reshape(flamingo_orig_dim, flamingo_red_dim)
        print(f"Flamingo projection: [{flamingo_orig_dim}, {flamingo_red_dim}]")

    # fused_projection: Vt[512, 1024] stored row-major
    fused_proj = None
    if fused_proj_blob:
        fused_proj = np.frombuffer(fused_proj_blob[0], dtype=np.float32).copy()
        fused_proj = fused_proj.reshape(512, 1024)
        print(f"Fused projection: [512, 1024]")

    print()
    print("=" * 80)

    mulan_sims = []
    flamingo_sims = []
    fused_sims = []

    for track in benchmark["tracks"]:
        artist = track["artist"]
        title = track["title"]
        print(f"\n{artist} - {title}")
        print(f"  Path: {track['path']}")

        # Find matching track in desktop DB
        matches = match_track_in_db(artist, title, db_path)
        if not matches:
            print("  NO MATCH in desktop DB")
            continue

        best_id, db_artist, db_title, score = matches[0]
        if score < 1.0:
            print(f"  Matched: {db_artist} - {db_title} (score={score:.2f})")
        else:
            print(f"  Matched: id={best_id}")

        # MuLan comparison
        if track.get("mulan") and track["mulan"].get("embedding"):
            phone_mulan = np.array(track["mulan"]["embedding"], dtype=np.float32)

            db_mulans = load_db_embeddings(db_path, [best_id], "embeddings_mulan")
            if best_id in db_mulans:
                db_mulan = db_mulans[best_id]
                sim = cosine_sim(phone_mulan, db_mulan)
                mulan_sims.append(sim)
                print(f"  MuLan:    cosine={sim:.6f}  ({track['mulan']['timingMs']}ms, {track['mulan']['ep']})")
            else:
                print(f"  MuLan:    no desktop embedding for track {best_id}")
        else:
            print(f"  MuLan:    no phone embedding")

        # Flamingo comparison
        if track.get("flamingo") and track["flamingo"].get("embedding"):
            phone_flamingo = np.array(track["flamingo"]["embedding"], dtype=np.float32)
            phone_dim = len(phone_flamingo)

            db_flamingos = load_db_embeddings(db_path, [best_id], "embeddings_flamingo")
            if best_id in db_flamingos and flamingo_proj is not None:
                db_flamingo = db_flamingos[best_id]

                if phone_dim == flamingo_orig_dim:
                    # Project phone's 3584d → 512d using same projection as desktop
                    phone_reduced = phone_flamingo @ flamingo_proj  # [3584] @ [3584, 512] → [512]
                    phone_reduced = phone_reduced / (np.linalg.norm(phone_reduced) + 1e-10)
                    sim = cosine_sim(phone_reduced, db_flamingo)
                elif phone_dim == flamingo_red_dim:
                    # Already reduced (shouldn't happen with projector, but handle it)
                    sim = cosine_sim(phone_flamingo, db_flamingo)
                else:
                    print(f"  Flamingo: unexpected dim {phone_dim}")
                    continue

                flamingo_sims.append(sim)
                print(f"  Flamingo: cosine={sim:.6f}  ({track['flamingo']['timingMs']}ms, {track['flamingo']['ep']})")

                # Compute fused embedding too
                if fused_proj is not None and track.get("mulan") and track["mulan"].get("embedding"):
                    phone_mulan = np.array(track["mulan"]["embedding"], dtype=np.float32)
                    # Reduce flamingo to 512d
                    if phone_dim == flamingo_orig_dim:
                        fl_512 = phone_flamingo @ flamingo_proj
                        fl_512 = fl_512 / (np.linalg.norm(fl_512) + 1e-10)
                    else:
                        fl_512 = phone_flamingo

                    # Concat [mulan_512 | flamingo_512] → 1024d
                    concat = np.concatenate([phone_mulan, fl_512])
                    # Project 1024d → 512d
                    phone_fused = fused_proj @ concat  # [512, 1024] @ [1024] → [512]
                    phone_fused = phone_fused / (np.linalg.norm(phone_fused) + 1e-10)

                    # Compare against desktop fused
                    db_fuseds = load_db_embeddings(db_path, [best_id], "embeddings_fused")
                    if best_id in db_fuseds:
                        sim_fused = cosine_sim(phone_fused, db_fuseds[best_id])
                        fused_sims.append(sim_fused)
                        print(f"  Fused:    cosine={sim_fused:.6f}")
            else:
                print(f"  Flamingo: no desktop embedding for track {best_id}")
        else:
            print(f"  Flamingo: no phone embedding")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if mulan_sims:
        print(f"MuLan    ({len(mulan_sims)} tracks): "
              f"mean={np.mean(mulan_sims):.6f}, "
              f"min={np.min(mulan_sims):.6f}, "
              f"max={np.max(mulan_sims):.6f}")
    else:
        print("MuLan:    no comparisons")

    if flamingo_sims:
        print(f"Flamingo ({len(flamingo_sims)} tracks): "
              f"mean={np.mean(flamingo_sims):.6f}, "
              f"min={np.min(flamingo_sims):.6f}, "
              f"max={np.max(flamingo_sims):.6f}")
    else:
        print("Flamingo: no comparisons")

    if fused_sims:
        print(f"Fused    ({len(fused_sims)} tracks): "
              f"mean={np.mean(fused_sims):.6f}, "
              f"min={np.min(fused_sims):.6f}, "
              f"max={np.max(fused_sims):.6f}")
    else:
        print("Fused:    no comparisons")

    print()
    print("Expected baselines (from prior TFLite validation):")
    print("  MuLan:    ~0.982")
    print("  Flamingo: ~0.990")
    print()
    if mulan_sims and np.mean(mulan_sims) > 0.95:
        print("MuLan: PASS")
    elif mulan_sims:
        print("MuLan: FAIL (below 0.95 threshold)")

    if flamingo_sims and np.mean(flamingo_sims) > 0.95:
        print("Flamingo: PASS")
    elif flamingo_sims:
        print("Flamingo: FAIL (below 0.95 threshold)")

    if fused_sims and np.mean(fused_sims) > 0.90:
        print("Fused: PASS")
    elif fused_sims:
        print("Fused: FAIL (below 0.90 threshold)")


if __name__ == "__main__":
    main()
