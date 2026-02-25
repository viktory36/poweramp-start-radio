#!/usr/bin/env python3
"""Validate TFLite CLaMP3 models against desktop PyTorch DB.

Runs the full TFLite pipeline (MERT → CLaMP3 audio encoder) on real audio
files and compares the resulting embeddings against the desktop-generated
embeddings in the CLaMP3 database.

This validates that the TFLite conversion preserves embedding quality before
deploying to Android.

Usage:
    python scripts/validate_tflite_clamp3.py --db audit_raw_data/embeddings_clamp3.db \
        --music /home/v/testmusic --n 50
"""

import argparse
import logging
import sqlite3
import time
import unicodedata
from pathlib import Path

import numpy as np
import torchaudio

logger = logging.getLogger(__name__)

MERT_SR = 24000
WINDOW_SAMPLES = 5 * MERT_SR  # 120,000

MODELS_DIR = Path(__file__).parent.parent / "models"


def load_tflite_model(path):
    """Load a TFLite model via ai_edge_litert."""
    from ai_edge_litert import interpreter as tfl_interp

    interp = tfl_interp.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    return interp


def run_tflite(interp, *inputs):
    """Run TFLite inference with arbitrary inputs."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    for i, inp in enumerate(inputs):
        interp.set_tensor(input_details[i]['index'], inp)

    interp.invoke()
    return interp.get_tensor(output_details[0]['index'])


def load_db_embeddings(db_path):
    """Load all embeddings from the CLaMP3 database, keyed by metadata key."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("""
        SELECT t.metadata_key, e.embedding
        FROM tracks t JOIN embeddings_clamp3 e ON t.id = e.track_id
    """).fetchall()
    conn.close()

    embeddings = {}
    for key, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32)
        embeddings[key] = emb

    logger.info(f"Loaded {len(embeddings)} embeddings from DB")
    return embeddings


def _normalize_field(value: str) -> str:
    """Lowercase, strip, NFC-normalize, and remove pipe characters."""
    nfc = unicodedata.normalize('NFC', value)
    return unicodedata.normalize('NFC', nfc.lower().strip().replace("|", "/"))


def get_track_metadata(fpath):
    """Extract metadata key from audio file using mutagen (matches fingerprint.py)."""
    import mutagen

    f = mutagen.File(str(fpath), easy=True)
    if f is None:
        return None

    artist = _normalize_field((f.get('artist', ['']) or [''])[0])
    album = _normalize_field((f.get('album', ['']) or [''])[0])
    title = _normalize_field((f.get('title', ['']) or [''])[0])
    duration_ms = int(f.info.length * 1000) if f.info else 0
    duration_rounded = (duration_ms // 100) * 100

    return f"{artist}|{album}|{title}|{duration_rounded}"


def extract_mert_features_tflite(interp, fpath, max_duration=600):
    """Run MERT TFLite on audio file, returning [N, 768] feature array."""
    from transformers import Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=MERT_SR, padding_value=0.0,
        return_attention_mask=True, do_normalize=True,
    )

    waveform, sr = torchaudio.load(str(fpath))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != MERT_SR:
        waveform = torchaudio.transforms.Resample(sr, MERT_SR)(waveform)

    max_samples = max_duration * MERT_SR
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]

    wav_np = waveform.squeeze(0).numpy()
    wav = processor(
        wav_np, return_tensors="np",
        sampling_rate=MERT_SR, padding=True,
    ).input_values[0]  # [T] 1D normalized

    features = []
    for j in range(0, len(wav), WINDOW_SAMPLES):
        chunk = wav[j:j + WINDOW_SAMPLES]
        if len(chunk) < MERT_SR:
            continue
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))

        # Run MERT: [1, 1, 120000, 1] → [1, 768]
        inp = chunk.reshape(1, 1, -1, 1).astype(np.float32)
        feat = run_tflite(interp, inp)
        features.append(feat[0])

    return np.array(features) if features else None


def encode_clamp3_audio_tflite(interp, features):
    """Run CLaMP3 audio encoder TFLite, returning [768] embedding.

    Matches the desktop CLaMP3 pipeline:
    1. Prepend + append zero vector (CLaMP3 training convention)
    2. Segment into 128-frame windows
    3. Last segment uses the final 128 frames (may overlap with previous)
    4. Weight-average segments by valid frame count
    """
    MAX_WINDOWS = 128
    FEATURE_DIM = 768

    num_windows = len(features)
    if num_windows == 0:
        return None

    # Prepend + append zero vector (matching CLaMP3 training pipeline)
    zero_vec = np.zeros((1, FEATURE_DIM), dtype=np.float32)
    input_data = np.concatenate([zero_vec, features, zero_vec], axis=0)
    total_frames = len(input_data)

    # Build segment list (matching desktop: last segment = last 128 frames)
    seg_starts = list(range(0, total_frames, MAX_WINDOWS))
    if len(seg_starts) > 1 or total_frames > MAX_WINDOWS:
        seg_starts[-1] = max(0, total_frames - MAX_WINDOWS)

    # Compute weights matching desktop logic
    full_cnt = total_frames // MAX_WINDOWS
    remain = total_frames % MAX_WINDOWS
    if remain == 0:
        weights = [MAX_WINDOWS] * full_cnt
    else:
        weights = [MAX_WINDOWS] * full_cnt + [remain]

    sum_emb = np.zeros(FEATURE_DIM, dtype=np.float32)
    total_weight = 0

    for s, seg_start in enumerate(seg_starts):
        seg_end = min(seg_start + MAX_WINDOWS, total_frames)
        seg_data = input_data[seg_start:seg_end]
        seg_len = len(seg_data)

        # Build padded input
        audio_inputs = np.zeros((1, MAX_WINDOWS, FEATURE_DIM), dtype=np.float32)
        audio_masks = np.zeros((1, MAX_WINDOWS), dtype=np.float32)
        audio_inputs[0, :seg_len] = seg_data
        audio_masks[0, :seg_len] = 1.0

        # Run CLaMP3 audio: [1, 128, 768] + [1, 128] → [1, 768]
        out = run_tflite(interp, audio_inputs, audio_masks)
        w = weights[s]
        sum_emb += out[0] * w
        total_weight += w

    if total_weight == 0:
        return None

    emb = sum_emb / total_weight
    # L2 normalize
    norm = np.linalg.norm(emb)
    if norm > 1e-10:
        emb /= norm
    return emb


def main():
    parser = argparse.ArgumentParser(description="Validate TFLite CLaMP3 models")
    parser.add_argument("--db", required=True, help="Path to embeddings_clamp3.db")
    parser.add_argument("--music", required=True, help="Path to test music directory")
    parser.add_argument("--n", type=int, default=20, help="Number of tracks to test")
    parser.add_argument("--mert-model", default=None, help="Path to MERT TFLite model")
    parser.add_argument("--clamp3-model", default=None, help="Path to CLaMP3 audio TFLite model")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    suffix = "_fp16" if args.fp16 else ""
    mert_path = args.mert_model or str(MODELS_DIR / f"mert{suffix}.tflite")
    clamp3_path = args.clamp3_model or str(MODELS_DIR / f"clamp3_audio{suffix}.tflite")

    logger.info(f"Loading MERT model: {mert_path}")
    mert_interp = load_tflite_model(mert_path)

    logger.info(f"Loading CLaMP3 audio model: {clamp3_path}")
    clamp3_interp = load_tflite_model(clamp3_path)

    logger.info(f"Loading DB: {args.db}")
    db_embeddings = load_db_embeddings(args.db)

    music_dir = Path(args.music)
    audio_files = sorted(
        p for p in music_dir.iterdir()
        if p.suffix.lower() in ('.flac', '.mp3', '.wav', '.ogg', '.m4a')
    )[:args.n]

    logger.info(f"Testing {len(audio_files)} tracks...")

    cosines = []
    matched = 0
    skipped = 0

    for i, fpath in enumerate(audio_files):
        key = get_track_metadata(fpath)
        if key is None or key not in db_embeddings:
            logger.warning(f"[{i+1}] Skip (no DB match): {fpath.name}")
            skipped += 1
            continue

        db_emb = db_embeddings[key]

        t0 = time.perf_counter()
        features = extract_mert_features_tflite(mert_interp, fpath)
        mert_ms = (time.perf_counter() - t0) * 1000

        if features is None:
            logger.warning(f"[{i+1}] Skip (no features): {fpath.name}")
            skipped += 1
            continue

        t0 = time.perf_counter()
        tflite_emb = encode_clamp3_audio_tflite(clamp3_interp, features)
        clamp3_ms = (time.perf_counter() - t0) * 1000

        if tflite_emb is None:
            logger.warning(f"[{i+1}] Skip (encode failed): {fpath.name}")
            skipped += 1
            continue

        cosine = np.dot(tflite_emb, db_emb) / (
            np.linalg.norm(tflite_emb) * np.linalg.norm(db_emb)
        )
        cosines.append(cosine)
        matched += 1

        status = "OK" if cosine > 0.99 else "WARN" if cosine > 0.95 else "FAIL"
        logger.info(
            f"[{i+1}/{len(audio_files)}] {status} cos={cosine:.6f} "
            f"mert={mert_ms:.0f}ms clamp3={clamp3_ms:.0f}ms "
            f"({features.shape[0]} windows) {fpath.name}"
        )

    print("\n" + "=" * 60)
    print(f"Results: {matched} matched, {skipped} skipped")
    if cosines:
        arr = np.array(cosines)
        print(f"  Cosine: mean={arr.mean():.6f}, min={arr.min():.6f}, "
              f"max={arr.max():.6f}, std={arr.std():.6f}")
        print(f"  >0.99: {np.sum(arr > 0.99)}/{len(arr)} "
              f"({np.sum(arr > 0.99)/len(arr)*100:.1f}%)")
        print(f"  >0.95: {np.sum(arr > 0.95)}/{len(arr)} "
              f"({np.sum(arr > 0.95)/len(arr)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    import torch
    main()
