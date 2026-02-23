#!/usr/bin/env python3
"""
Generate CLaMP3 audio embeddings for a full music library.

Two-phase pipeline:
  Phase 1: Extract MERT features from audio → cache as .npy (~0.7s/track on GPU)
  Phase 2: Encode MERT features via CLaMP3 audio encoder → SQLite DB (~0.02s/track)

Self-contained — no dependency on the CLaMP3 source repo.
Models auto-download from HuggingFace on first run.

Usage:
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db --phase 1
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db --phase 2
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db --max-duration 600 --batch-size 8
"""

import argparse
import gc
import json
import os
import re
import sqlite3
import struct
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoModel, BertConfig, BertModel, Wav2Vec2FeatureExtractor


# ─── CLaMP3 constants (from CLaMP3/code/config.py) ───────────────────────────

MAX_AUDIO_LENGTH = 128       # Max sequence length for audio encoder
AUDIO_HIDDEN_SIZE = 768      # BertModel hidden size
AUDIO_NUM_LAYERS = 12        # BertModel layers
CLAMP3_HIDDEN_SIZE = 768     # Output embedding dim

# MERT audio params
MERT_SR = 24000              # 24kHz target sample rate
WINDOW_SEC = 5               # 5s non-overlapping windows
WINDOW_SAMPLES = WINDOW_SEC * MERT_SR  # 120,000

# CLaMP3 weights filename (deterministic from CLaMP3 config)
CLAMP3_WEIGHTS_FILENAME = (
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base"
    "_t_length_128_a_size_768_a_layers_12_a_length_128"
    "_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
)

AUDIO_EXTENSIONS = {'.flac', '.mp3', '.opus', '.wav', '.m4a', '.ogg', '.wma', '.aac'}


# ─── CLaMP3 audio encoder (vendored from CLaMP3/code/utils.py) ───────────────

class CLaMP3AudioEncoder(torch.nn.Module):
    """Minimal CLaMP3 audio encoder: BertModel + linear projection.

    Extracts from the full CLaMP3Model only the audio pathway needed for
    generating audio embeddings from MERT features.
    """

    def __init__(self):
        super().__init__()
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE // 64,  # 12 heads
            intermediate_size=AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=MAX_AUDIO_LENGTH,
        )
        self.audio_model = BertModel(audio_config)
        self.audio_proj = torch.nn.Linear(AUDIO_HIDDEN_SIZE, CLAMP3_HIDDEN_SIZE)

    @torch.no_grad()
    def encode(self, audio_inputs, audio_masks):
        """Encode MERT features → 768d embedding (not L2-normalized).

        Args:
            audio_inputs: [B, seq_len, 768] MERT features (zero-padded to MAX_AUDIO_LENGTH)
            audio_masks:  [B, seq_len] attention masks (1=real, 0=padding)

        Returns:
            [B, 768] projected features
        """
        features = self.audio_model(
            inputs_embeds=audio_inputs,
            attention_mask=audio_masks,
        )['last_hidden_state']
        # Masked average pooling (matches CLaMP3Model.avg_pooling)
        masks = audio_masks.unsqueeze(-1).to(features.device)
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1)
        return self.audio_proj(pooled)

    @classmethod
    def from_clamp3_checkpoint(cls, weights_path, device='cpu'):
        """Load only audio_model + audio_proj from a full CLaMP3 checkpoint."""
        model = cls()
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        full_state = checkpoint['model']

        # Extract only audio keys
        audio_state = {}
        for k, v in full_state.items():
            if k.startswith('audio_model.') or k.startswith('audio_proj.'):
                audio_state[k] = v

        model.load_state_dict(audio_state)
        model.to(device).eval()
        print(f"CLaMP3 audio encoder loaded (epoch {checkpoint.get('epoch', '?')})")
        return model


# ─── Metadata extraction (inline from fingerprint.py) ────────────────────────

def _normalize_field(value):
    """Lowercase, strip, NFC-normalize, and remove pipe characters."""
    nfc = unicodedata.normalize('NFC', value)
    return unicodedata.normalize('NFC', nfc.lower().strip().replace("|", "/"))


def make_metadata_key(artist, album, title, duration_ms):
    """Create metadata key: 'artist|album|title|duration_rounded'."""
    a = _normalize_field(artist or "")
    b = _normalize_field(album or "")
    t = _normalize_field(title or "")
    dur = (duration_ms // 100) * 100
    return f"{a}|{b}|{t}|{dur}"


def make_filename_key(file_path):
    """Create normalized filename key for fallback matching."""
    name = Path(file_path).stem.lower()
    name = re.sub(r'\s*[\(\[].*?[\)\]]', '', name)
    name = re.sub(r'^\d+[\.\-\s]+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return unicodedata.normalize('NFC', name)


def extract_metadata(file_path):
    """Extract artist, album, title, duration_ms from audio file using mutagen.

    Returns:
        (artist, album, title, duration_ms) tuple
    """
    try:
        import mutagen
    except ImportError:
        return None, None, Path(file_path).stem, 0

    try:
        audio = mutagen.File(str(file_path), easy=True)
        if audio is None:
            return None, None, Path(file_path).stem, 0

        duration_ms = int((audio.info.length if audio.info else 0) * 1000)

        def get_tag(tags):
            for tag in tags:
                if tag in audio:
                    val = audio[tag]
                    if isinstance(val, list) and val:
                        return str(val[0])
                    elif val:
                        return str(val)
            return None

        artist = get_tag(['artist', 'albumartist', 'performer'])
        album = get_tag(['album'])
        title = get_tag(['title']) or Path(file_path).stem
        return artist, album, title, duration_ms
    except Exception:
        return None, None, Path(file_path).stem, 0


# ─── Database helpers ─────────────────────────────────────────────────────────

def float_list_to_blob(floats):
    """Convert list of floats to binary blob (float32)."""
    return struct.pack(f'{len(floats)}f', *floats)


DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metadata_key TEXT NOT NULL,
    filename_key TEXT NOT NULL,
    artist TEXT,
    album TEXT,
    title TEXT,
    duration_ms INTEGER,
    file_path TEXT NOT NULL,
    cluster_id INTEGER,
    source TEXT DEFAULT 'desktop',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tracks_metadata_key ON tracks(metadata_key);
CREATE INDEX IF NOT EXISTS idx_tracks_filename_key ON tracks(filename_key);
CREATE INDEX IF NOT EXISTS idx_tracks_file_path ON tracks(file_path);

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS binary_data (
    key TEXT PRIMARY KEY,
    data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings_clamp3 (
    track_id INTEGER PRIMARY KEY,
    embedding BLOB NOT NULL,
    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
);
"""


def init_db(db_path):
    """Create/open SQLite DB with the embedding schema."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(DB_SCHEMA)
    conn.commit()
    return conn


def get_existing_paths_with_embeddings(conn):
    """Return set of file_paths that already have CLaMP3 embeddings."""
    rows = conn.execute(
        "SELECT t.file_path FROM tracks t "
        "INNER JOIN embeddings_clamp3 e ON t.id = e.track_id"
    ).fetchall()
    return {row['file_path'] for row in rows}


# ─── File discovery ───────────────────────────────────────────────────────────

def discover_audio_files(music_dir):
    """Recursively find all audio files."""
    files = []
    for root, dirs, filenames in os.walk(music_dir):
        for f in sorted(filenames):
            if Path(f).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(Path(root) / f)
    return files


def make_cache_key(file_path, music_dir):
    """Create a flat cache key from a file path relative to music_dir."""
    relative = str(file_path.relative_to(music_dir)).replace("\\", "/")
    return relative.replace("/", "__")


# ─── Phase 1: MERT feature extraction ────────────────────────────────────────

def phase1_mert(music_dir, cache_dir, max_duration, batch_size):
    """Extract MERT features from all audio files, cached as .npy.

    Pipeline (matches official CLaMP3 exactly):
      Audio → 24kHz mono → Wav2Vec2FeatureExtractor(do_normalize=True)
        → 5s non-overlapping windows (discard < 1s, pad last to 5s)
        → MERT-v1-95M: layer=None, reduction="mean" → [L, chunks, 768]
        → Mean over layers → [1, chunks, 768] → save .npy
    """
    print("=" * 70)
    print("PHASE 1: MERT Feature Extraction")
    print("=" * 70)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    music_dir = Path(music_dir)

    # Discover all audio files
    all_files = discover_audio_files(music_dir)
    print(f"Found {len(all_files)} audio files")

    # Load or create manifest (maps cache_key → original absolute path)
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    # Find files needing processing
    to_process = []
    for fpath in all_files:
        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")
        if not npy_path.exists():
            to_process.append(fpath)
        # Always update manifest
        manifest[cache_key] = str(fpath)

    # Save manifest (even if nothing to process — captures new files)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=0)

    if not to_process:
        print(f"All {len(all_files)} files already cached. Skipping Phase 1.")
        return

    print(f"Need to process {len(to_process)} files "
          f"({len(all_files) - len(to_process)} cached)")

    # Load MERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MERT-v1-95M on {device}...")

    processor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=MERT_SR, padding_value=0.0,
        return_attention_mask=True, do_normalize=True,
    )
    mert_model = AutoModel.from_pretrained(
        "m-a-p/MERT-v1-95M", trust_remote_code=True
    )
    mert_model.to(device).eval()
    for param in mert_model.parameters():
        param.requires_grad = False
    print("MERT loaded")

    t0 = time.time()
    success = 0
    fail = 0

    for i, fpath in enumerate(to_process):
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(to_process) - i) / 60
            print(f"  [{i}/{len(to_process)}] {success} ok, {fail} fail, "
                  f"{rate:.1f}s/track, ETA {eta:.0f}min")

        if i > 0 and i % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")

        try:
            # Load and resample to 24kHz mono (matches CLaMP3 MERT_utils.load_audio)
            waveform, sr = torchaudio.load(str(fpath))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != MERT_SR:
                waveform = torchaudio.transforms.Resample(sr, MERT_SR)(waveform)

            # Cap duration
            max_samples = max_duration * MERT_SR
            if waveform.shape[-1] > max_samples:
                waveform = waveform[:, :max_samples]

            # Normalize with Wav2Vec2FeatureExtractor (matching CLaMP3 pipeline)
            wav_np = waveform.squeeze(0).numpy()
            wav = processor(
                wav_np, return_tensors="pt",
                sampling_rate=MERT_SR, padding=True,
            ).input_values[0]  # [T] 1D normalized

            # Split into 5s non-overlapping windows
            chunks = []
            for j in range(0, len(wav), WINDOW_SAMPLES):
                chunk = wav[j:j + WINDOW_SAMPLES]
                if len(chunk) < MERT_SR:  # < 1s, discard
                    continue
                if len(chunk) < WINDOW_SAMPLES:  # Pad last window to 5s
                    chunk = torch.nn.functional.pad(
                        chunk, (0, WINDOW_SAMPLES - len(chunk))
                    )
                chunks.append(chunk)

            if not chunks:
                fail += 1
                continue

            # Batch process through MERT
            features_list = []
            for b_start in range(0, len(chunks), batch_size):
                batch = torch.stack(
                    chunks[b_start:b_start + batch_size]
                ).to(device)
                with torch.no_grad():
                    out = mert_model(
                        batch, output_hidden_states=True
                    ).hidden_states
                    out = torch.stack(out)   # [L, B, T_hidden, H]
                    out = out.mean(-2)       # [L, B, H] — mean over time
                features_list.append(out.cpu())

            # Concatenate batches → [L, total_chunks, H]
            all_features = torch.cat(features_list, dim=1)
            # Mean over layers (matching CLaMP3 --mean_features)
            all_features = all_features.mean(dim=0, keepdim=True)  # [1, chunks, H]

            np.save(str(npy_path), all_features.numpy())
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  FAIL: {fpath.name[:60]}: {e}")

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {success} ok, {fail} fail in {elapsed:.0f}s "
          f"({elapsed / max(1, success):.1f}s/track)")

    # Cleanup — free VRAM for Phase 2
    del mert_model
    torch.cuda.empty_cache()
    gc.collect()


# ─── Phase 2: CLaMP3 encoding + DB storage ───────────────────────────────────

def phase2_clamp3(music_dir, cache_dir, db_path):
    """Encode cached MERT features with CLaMP3 and store in SQLite.

    Pipeline (matches official CLaMP3 exactly):
      Load .npy → [chunks, 768] → prepend/append zero vector
        → segment into windows of 128 (last = last 128 frames)
        → BertModel → avg_pooling → audio_proj
        → weighted average of segments → L2 normalize → 768d
        → store in SQLite with track metadata
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CLaMP3 Encoding → SQLite")
    print("=" * 70)

    cache_dir = Path(cache_dir)
    music_dir = Path(music_dir)
    db_path = Path(db_path)

    # Load manifest
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: No manifest.json found in cache dir. Run Phase 1 first.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find all cached .npy files
    npy_files = sorted(cache_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} cached MERT feature files")

    if not npy_files:
        print("No .npy files found. Run Phase 1 first.")
        return

    # Init DB and check existing embeddings
    conn = init_db(db_path)
    existing = get_existing_paths_with_embeddings(conn)

    # Build lookup for orphan tracks (exist in DB without embeddings)
    orphan_tracks = {}
    rows = conn.execute("SELECT id, file_path FROM tracks").fetchall()
    for row in rows:
        if row['file_path'] not in existing:
            orphan_tracks[row['file_path']] = row['id']

    # Filter to files needing processing
    to_process = []
    for npy_path in npy_files:
        cache_key = npy_path.stem
        original_path = manifest.get(cache_key)
        if original_path and original_path not in existing:
            to_process.append((npy_path, cache_key, original_path))

    print(f"Need to encode {len(to_process)} tracks "
          f"({len(npy_files) - len(to_process)} already in DB)")

    if not to_process:
        print("All tracks already have embeddings. Done.")
        conn.close()
        return

    # Download CLaMP3 weights if needed
    from huggingface_hub import hf_hub_download
    print("Loading CLaMP3 weights...")
    weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)

    # Load CLaMP3 audio encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = CLaMP3AudioEncoder.from_clamp3_checkpoint(weights_path, device=device)

    # Set metadata
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("model", "clamp3"))
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("source_path", str(music_dir)))
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
        ("embedding_dim", "768"))
    conn.commit()

    t0 = time.time()
    success = 0
    fail = 0

    for i, (npy_path, cache_key, original_path) in enumerate(to_process):
        if i > 0 and i % 500 == 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(to_process) - i) / 60
            print(f"  [{i}/{len(to_process)}] {success} ok, {fail} fail, "
                  f"{rate:.3f}s/track, ETA {eta:.0f}min")
            conn.commit()

        try:
            # Load MERT features
            input_data = np.load(str(npy_path))
            input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = input_data.reshape(-1, input_data.size(-1))  # [chunks, 768]

            # Prepend + append zero vector (matching CLaMP3 training pipeline)
            zero_vec = torch.zeros((1, input_data.size(-1)))
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)

            # Segment into windows of MAX_AUDIO_LENGTH (128)
            segment_list = []
            for j in range(0, len(input_data), MAX_AUDIO_LENGTH):
                segment_list.append(input_data[j:j + MAX_AUDIO_LENGTH])
            # Last segment = last 128 frames (overlaps previous if needed)
            segment_list[-1] = input_data[-MAX_AUDIO_LENGTH:]

            # Encode each segment
            hidden_states_list = []
            for seg in segment_list:
                actual_len = seg.size(0)
                masks = torch.zeros(MAX_AUDIO_LENGTH)
                masks[:actual_len] = 1.0
                if actual_len < MAX_AUDIO_LENGTH:
                    pad = torch.zeros(
                        (MAX_AUDIO_LENGTH - actual_len, AUDIO_HIDDEN_SIZE)
                    )
                    seg = torch.cat((seg, pad), 0)

                feat = encoder.encode(
                    seg.unsqueeze(0).to(device),
                    masks.unsqueeze(0).to(device),
                )
                hidden_states_list.append(feat)

            # Weighted average of segments (weight = actual frame count)
            total_frames = len(input_data)
            full_cnt = total_frames // MAX_AUDIO_LENGTH
            remain = total_frames % MAX_AUDIO_LENGTH
            if remain == 0:
                weights = torch.tensor(
                    [MAX_AUDIO_LENGTH] * full_cnt, device=device
                ).view(-1, 1)
            else:
                weights = torch.tensor(
                    [MAX_AUDIO_LENGTH] * full_cnt + [remain], device=device
                ).view(-1, 1)

            stacked = torch.cat(hidden_states_list, 0)
            emb = (stacked * weights).sum(0) / weights.sum()
            emb = emb / emb.norm()  # L2 normalize
            emb_list = emb.cpu().tolist()

            # Extract metadata from original file
            artist, album, title, duration_ms = extract_metadata(original_path)
            metadata_key = make_metadata_key(artist, album, title, duration_ms)
            filename_key = make_filename_key(original_path)

            # Get or create track in DB (handles interrupted runs)
            if original_path in orphan_tracks:
                track_id = orphan_tracks[original_path]
            else:
                cursor = conn.execute(
                    "INSERT INTO tracks (metadata_key, filename_key, artist, "
                    "album, title, duration_ms, file_path, source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (metadata_key, filename_key, artist, album, title,
                     duration_ms, original_path, 'desktop')
                )
                track_id = cursor.lastrowid

            conn.execute(
                "INSERT OR REPLACE INTO embeddings_clamp3 "
                "(track_id, embedding) VALUES (?, ?)",
                (track_id, float_list_to_blob(emb_list))
            )
            success += 1

        except Exception as e:
            fail += 1
            if fail <= 20:
                print(f"  FAIL: {cache_key[:60]}: {e}")

    conn.commit()
    elapsed = time.time() - t0

    # Report
    total_tracks = conn.execute(
        "SELECT COUNT(*) as c FROM tracks"
    ).fetchone()['c']
    total_embs = conn.execute(
        "SELECT COUNT(*) as c FROM embeddings_clamp3"
    ).fetchone()['c']
    print(f"\nPhase 2 complete: {success} ok, {fail} fail in {elapsed:.0f}s "
          f"({elapsed / max(1, success):.3f}s/track)")
    print(f"Database: {total_tracks} tracks, {total_embs} CLaMP3 embeddings")
    print(f"Saved to {db_path}")

    conn.close()
    del encoder
    torch.cuda.empty_cache()
    gc.collect()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate CLaMP3 audio embeddings for a music library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python generate_clamp3_embeddings.py C:\\backups\\Music -o embeddings_clamp3.db
  python generate_clamp3_embeddings.py /path/to/music -o emb.db --phase 1
  python generate_clamp3_embeddings.py /path/to/music -o emb.db --phase 2
  python generate_clamp3_embeddings.py /path/to/music -o emb.db --max-duration 600 --batch-size 8
        """
    )
    parser.add_argument(
        "music_dir",
        help="Root directory of the music library")
    parser.add_argument(
        "-o", "--output", default="embeddings_clamp3.db",
        help="Output SQLite database path (default: embeddings_clamp3.db)")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for MERT feature cache "
             "(default: mert_cache/ next to output DB)")
    parser.add_argument(
        "--phase", choices=["1", "2", "both"], default="both",
        help="Which phase to run (default: both)")
    parser.add_argument(
        "--max-duration", type=int, default=600,
        help="Max audio duration in seconds (default: 600 = 10 min)")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="MERT chunk batch size for GPU throughput (default: 8)")

    args = parser.parse_args()

    music_dir = Path(args.music_dir).resolve()
    if not music_dir.is_dir():
        print(f"ERROR: Music directory not found: {music_dir}")
        return 1

    db_path = Path(args.output).resolve()
    if args.cache_dir:
        cache_dir = Path(args.cache_dir).resolve()
    else:
        cache_dir = db_path.parent / "mert_cache"

    print(f"Music directory: {music_dir}")
    print(f"Output database: {db_path}")
    print(f"Cache directory: {cache_dir}")
    print(f"Max duration: {args.max_duration}s, Batch size: {args.batch_size}")
    print()

    if args.phase in ("1", "both"):
        phase1_mert(music_dir, cache_dir, args.max_duration, args.batch_size)

    if args.phase in ("2", "both"):
        phase2_clamp3(music_dir, cache_dir, db_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
