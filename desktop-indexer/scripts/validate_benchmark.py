#!/usr/bin/env python3
"""Validate on-device CLaMP3 benchmark embeddings against the desktop database.

Usage:
    python scripts/validate_benchmark.py <benchmark_json> <desktop_db>
    python scripts/validate_benchmark.py /path/to/benchmark_results-clamp3.json audit_raw_data/embeddings_clamp3.db
"""
import argparse
import json
import sqlite3
import sys

import numpy as np


def load_embedding_from_blob(blob: bytes) -> np.ndarray:
    """Decode a 768-dim embedding from SQLite blob (always stored as float32)."""
    return np.frombuffer(blob, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="Path to benchmark_results-clamp3.json")
    parser.add_argument("db", help="Path to desktop embeddings_clamp3.db")
    args = parser.parse_args()

    with open(args.benchmark) as f:
        benchmark = json.load(f)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    # Header
    print(f"Device: {benchmark['device']} ({benchmark['soc']})")
    print(f"Android: {benchmark['androidVersion']}")
    print(f"MERT: {benchmark['mertModel']} ({benchmark['mertAccelerator']}, load {benchmark['mertLoadMs']}ms)")
    print(f"CLaMP3: {benchmark['clamp3Model']} ({benchmark['clamp3Accelerator']}, load {benchmark['clamp3LoadMs']}ms)")
    print(f"Runtime: {benchmark['runtime']}")
    print()

    tracks = benchmark["tracks"]
    device_embs = []
    desktop_embs = []
    names = []
    cosines = []

    for i, track in enumerate(tracks):
        artist, album, title = track["artist"], track["album"], track["title"]
        device_emb = np.array(track["clamp3"]["embedding"], dtype=np.float32)
        assert len(device_emb) == 768

        # Look up in desktop DB (exact match, then artist+title fallback)
        for query, params in [
            ("t.artist = ? AND t.album = ? AND t.title = ?", (artist, album, title)),
            ("t.artist = ? AND t.title = ?", (artist, title)),
            ("LOWER(t.artist) = LOWER(?) AND LOWER(t.title) = LOWER(?)", (artist, title)),
        ]:
            cur.execute(
                f"SELECT t.id, e.embedding "
                f"FROM tracks t JOIN embeddings_clamp3 e ON t.id = e.track_id "
                f"WHERE {query}", params
            )
            rows = cur.fetchall()
            if rows:
                break

        if not rows:
            print(f"Track {i+1}: {artist} - {title}")
            print(f"  NOT FOUND in desktop DB\n")
            continue

        db_id, blob = rows[0]
        desktop_emb = load_embedding_from_blob(blob)
        assert len(desktop_emb) == 768

        cos = cosine_similarity(device_emb, desktop_emb)
        cosines.append(cos)
        device_embs.append(device_emb)
        desktop_embs.append(desktop_emb)
        names.append(f"{artist} - {title}")

        timing = track.get("timing", {})
        print(f"Track {i+1}: {artist} - {title}")
        print(f"  Album: {album} | DB id: {db_id}")
        print(f"  Duration: {track.get('durationS', '?')}s")
        if timing:
            print(f"  Timing: decode {timing.get('decodingMs', '?')}ms, "
                  f"resample {timing.get('resamplingMs', '?')}ms, "
                  f"MERT {timing.get('mertMs', '?')}ms, "
                  f"CLaMP3 {timing.get('clamp3Ms', '?')}ms, "
                  f"total {timing.get('totalMs', '?')}ms")
        print(f"  Device  norm: {np.linalg.norm(device_emb):.6f}")
        print(f"  Desktop norm: {np.linalg.norm(desktop_emb):.6f}")
        print(f"  Cosine similarity: {cos:.6f}")
        print()

    conn.close()

    if len(cosines) < 2:
        print("Not enough matched tracks for analysis.")
        return

    # Pairwise analysis
    n = len(device_embs)
    print("=" * 70)
    print("PAIRWISE COSINE SIMILARITY (on-device)")
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.dot(device_embs[i], device_embs[j])
            print(f"  {names[i][:35]:35s} vs {names[j][:35]:35s}: {cos:.4f}")

    print()
    print("PAIRWISE COSINE SIMILARITY (desktop)")
    for i in range(n):
        for j in range(i + 1, n):
            cos = np.dot(desktop_embs[i], desktop_embs[j])
            print(f"  {names[i][:35]:35s} vs {names[j][:35]:35s}: {cos:.4f}")

    # Summary
    print()
    print("=" * 70)
    print(f"SUMMARY ({len(cosines)}/{len(tracks)} tracks matched)")
    print(f"  Per-track cosine vs desktop:")
    for name, cos in zip(names, cosines):
        print(f"    {name}: {cos:.6f}")
    print()
    print(f"  Mean cosine:    {np.mean(cosines):.6f}")
    print(f"  Min cosine:     {np.min(cosines):.6f}")
    print(f"  Max cosine:     {np.max(cosines):.6f}")
    print(f"  Std dev:        {np.std(cosines):.6f}")
    print()

    # Device embedding collapse check
    device_pairwise = []
    desktop_pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            device_pairwise.append(np.dot(device_embs[i], device_embs[j]))
            desktop_pairwise.append(np.dot(desktop_embs[i], desktop_embs[j]))

    print(f"  Mean pairwise cosine (device):  {np.mean(device_pairwise):.4f}")
    print(f"  Mean pairwise cosine (desktop): {np.mean(desktop_pairwise):.4f}")
    print()

    if np.mean(cosines) >= 0.98:
        print("PASS: On-device embeddings closely match desktop (mean >= 0.98)")
    elif np.mean(cosines) >= 0.95:
        print("ACCEPTABLE: Minor deviation from desktop (mean >= 0.95)")
    elif np.mean(device_pairwise) > 0.90:
        print("FAIL: On-device embeddings have COLLAPSED — all tracks produce nearly")
        print("      identical embeddings regardless of content. The GPU inference")
        print("      pipeline is not working correctly.")
        print(f"      (device pairwise mean {np.mean(device_pairwise):.4f} >> "
              f"desktop pairwise mean {np.mean(desktop_pairwise):.4f})")
    else:
        print(f"FAIL: On-device embeddings diverge from desktop (mean cosine {np.mean(cosines):.4f})")

    sys.exit(0 if np.mean(cosines) >= 0.95 else 1)


if __name__ == "__main__":
    main()
