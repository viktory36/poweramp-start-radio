#!/usr/bin/env python3
"""Validate on-device CLaMP3 benchmark embeddings against desktop DB.

Pull benchmark_results.json from phone and compare each track's embedding
against the desktop-generated embeddings in the CLaMP3 database.

Usage:
    adb pull /data/data/com.powerampstartradio/files/benchmark_results.json
    python scripts/validate_benchmark.py benchmark_results.json \
        --db audit_raw_data/embeddings_clamp3.db
"""

import argparse
import json
import sqlite3
import unicodedata

import numpy as np


def _normalize_field(value: str) -> str:
    """Lowercase, strip, NFC-normalize, remove pipe chars (matches fingerprint.py)."""
    nfc = unicodedata.normalize('NFC', value)
    return unicodedata.normalize('NFC', nfc.lower().strip().replace("|", "/"))


def load_benchmark(path: str) -> dict:
    """Load benchmark_results.json from the phone (handles UTF-16/BOM)."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8")
    else:
        text = raw.decode("utf-8")
    text = text.lstrip("\ufeff")
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Validate benchmark embeddings vs desktop DB")
    parser.add_argument("benchmark_json", help="Path to benchmark_results.json from phone")
    parser.add_argument("--db", required=True, help="Path to embeddings_clamp3.db")
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_json)

    conn = sqlite3.connect(args.db)
    db_rows = conn.execute("""
        SELECT t.metadata_key, e.embedding, e.precision
        FROM tracks t JOIN embeddings_clamp3 e ON t.id = e.track_id
    """).fetchall()

    db_embeddings = {}
    db_precision = {}
    for key, blob, prec in db_rows:
        db_embeddings[key] = np.frombuffer(blob, dtype=np.float32)
        db_precision[key] = prec
    conn.close()

    print(f"DB: {len(db_embeddings)} embeddings loaded")
    print(f"Device: {benchmark.get('device', '?')}")
    print(f"SOC: {benchmark.get('soc', '?')}")
    print(f"CLaMP3 EP: {benchmark.get('clamp3Ep', '?')}")
    print()

    cosines = []
    for track in benchmark['tracks']:
        artist = _normalize_field(track.get('artist', ''))
        album = _normalize_field(track.get('album', ''))
        title = _normalize_field(track.get('title', ''))
        dur_ms = track.get('durationMs', 0)
        dur_rounded = (dur_ms // 100) * 100

        key = f"{artist}|{album}|{title}|{dur_rounded}"

        clamp3 = track.get('clamp3')
        if clamp3 is None:
            print(f"SKIP (no embedding): {track['artist']} - {track['title']}")
            continue

        phone_emb = np.array(clamp3['embedding'], dtype=np.float32)

        # Try exact key match first, then fuzzy
        db_emb = db_embeddings.get(key)
        matched_key = key
        match_type = "exact"

        if db_emb is None:
            # Try without duration (Poweramp duration can differ from file metadata)
            for db_key in db_embeddings:
                parts = db_key.split('|')
                if len(parts) >= 3 and parts[0] == artist and parts[2] == title:
                    db_emb = db_embeddings[db_key]
                    matched_key = db_key
                    match_type = "artist+title"
                    break

        if db_emb is None:
            # Try title substring
            for db_key in db_embeddings:
                parts = db_key.split('|')
                if len(parts) >= 3 and parts[0] == artist and title in parts[2]:
                    db_emb = db_embeddings[db_key]
                    matched_key = db_key
                    match_type = "fuzzy"
                    break

        if db_emb is None:
            print(f"NO MATCH: {track['artist']} - {track['title']}")
            print(f"  Tried key: {key}")
            continue

        cosine = float(np.dot(phone_emb, db_emb) / (
            np.linalg.norm(phone_emb) * np.linalg.norm(db_emb) + 1e-10
        ))
        cosines.append(cosine)

        prec = db_precision.get(matched_key, '?')
        status = "OK" if cosine > 0.99 else "WARN" if cosine > 0.95 else "FAIL"
        timing = clamp3.get('timingMs', '?')

        print(f"{status} cos={cosine:.6f} [{prec}] {match_type:12s} "
              f"{timing}ms  {track['artist']} - {track['title']}")

    print()
    print("=" * 60)
    if cosines:
        arr = np.array(cosines)
        print(f"Results: {len(cosines)} tracks compared")
        print(f"  Cosine: mean={arr.mean():.6f}, min={arr.min():.6f}, max={arr.max():.6f}")
        print(f"  >0.99: {np.sum(arr > 0.99)}/{len(arr)}")
        print(f"  >0.95: {np.sum(arr > 0.95)}/{len(arr)}")
    else:
        print("No tracks matched!")
    print("=" * 60)


if __name__ == "__main__":
    main()
