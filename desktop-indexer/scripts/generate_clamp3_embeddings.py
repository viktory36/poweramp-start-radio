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
  python generate_clamp3_embeddings.py /path/to/music -o embeddings_clamp3.db --fp16
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


def _read_manifest_entry(entry):
    """Read manifest entry, handling old (string) and new (dict) formats."""
    if isinstance(entry, str):
        return entry, 'fp32'
    return entry['path'], entry['precision']


# ─── Phase 1: MERT feature extraction ────────────────────────────────────────

def _load_audio_chunks(fpath, processor, max_duration):
    """CPU-bound: load audio, resample, normalize, split into 5s chunks.

    Runs in a background thread to overlap with GPU inference.
    Falls back to soundfile when torchaudio fails (common with MP3s that
    have minor frame-level corruption — playable everywhere but rejected
    by torchaudio's strict FFmpeg decoder).

    Returns:
        (chunks, num_samples) where chunks is a list of [WINDOW_SAMPLES] tensors,
        or raises on failure.
    """
    try:
        waveform, sr = torchaudio.load(str(fpath))
    except Exception:
        # Fallback: soundfile/mpg123 is more forgiving with corrupt frames
        import soundfile as sf
        import soxr
        data, sr = sf.read(str(fpath), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != MERT_SR:
            data = soxr.resample(data, sr, MERT_SR)
        waveform = torch.from_numpy(data).unsqueeze(0)
        sr = MERT_SR  # already resampled

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != MERT_SR:
        waveform = torchaudio.transforms.Resample(sr, MERT_SR)(waveform)

    max_samples = max_duration * MERT_SR
    if waveform.shape[-1] > max_samples:
        waveform = waveform[:, :max_samples]

    wav_np = waveform.squeeze(0).numpy()
    wav = processor(
        wav_np, return_tensors="pt",
        sampling_rate=MERT_SR, padding=True,
    ).input_values[0]  # [T] 1D normalized

    chunks = []
    for j in range(0, len(wav), WINDOW_SAMPLES):
        chunk = wav[j:j + WINDOW_SAMPLES]
        if len(chunk) < MERT_SR:
            continue
        if len(chunk) < WINDOW_SAMPLES:
            chunk = torch.nn.functional.pad(
                chunk, (0, WINDOW_SAMPLES - len(chunk))
            )
        chunks.append(chunk)

    return chunks, waveform.shape[-1]


def phase1_mert(music_dir, cache_dir, max_duration, batch_size, verbose=False,
                fp16=False):
    """Extract MERT features from all audio files, cached as .npy.

    Pipeline (matches official CLaMP3 exactly):
      Audio → 24kHz mono → Wav2Vec2FeatureExtractor(do_normalize=True)
        → 5s non-overlapping windows (discard < 1s, pad last to 5s)
        → MERT-v1-95M: layer=None, reduction="mean" → [L, chunks, 768]
        → Mean over layers → [1, chunks, 768] → save .npy

    CPU audio decoding runs in a background thread, overlapping with GPU inference.
    When fp16=True, MERT runs in half precision (halves VRAM, ~2x faster).
    Output is cast back to float32 before saving for Phase 2 compatibility.
    """
    from concurrent.futures import ThreadPoolExecutor

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
    precision_str = 'fp16' if fp16 else 'fp32'
    to_process = []
    for fpath in all_files:
        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")
        if not npy_path.exists():
            to_process.append(fpath)
            # New files get current precision (written after successful .npy save)
        else:
            # Existing .npy — upgrade old manifest format, preserve precision
            existing = manifest.get(cache_key)
            if isinstance(existing, str):
                manifest[cache_key] = {'path': existing, 'precision': 'fp32'}
            elif existing is None:
                manifest[cache_key] = {'path': str(fpath), 'precision': 'fp32'}

    # Save manifest (even if nothing to process — captures format upgrades)
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
    if fp16:
        mert_model.half()
    mert_model = torch.compile(mert_model)
    print(f"MERT loaded{' (FP16)' if fp16 else ''}, compiling graph...")

    t0 = time.time()
    success = 0
    fail = 0

    # Prefetch audio decoding on CPU while GPU runs MERT on previous track
    pool = ThreadPoolExecutor(max_workers=2)
    prefetch = pool.submit(_load_audio_chunks,
                           to_process[0], processor, max_duration)

    for i, fpath in enumerate(to_process):
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(to_process) - i) / 60
            print(f"  [{i}/{len(to_process)}] {success} ok, {fail} fail, "
                  f"{rate:.1f}s/track, ETA {eta:.0f}min")

        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")

        # Get prefetched audio (or wait for it)
        try:
            chunks, num_samples = prefetch.result()
        except Exception as e:
            fail += 1
            if fail <= 20 or verbose:
                print(f"  FAIL [{i+1}/{len(to_process)}] {fpath.name[:60]}: {e}")
            # Submit next prefetch before continuing
            if i + 1 < len(to_process):
                prefetch = pool.submit(_load_audio_chunks,
                                       to_process[i + 1], processor,
                                       max_duration)
            continue

        # Submit next track's audio decode NOW (overlaps with GPU below)
        if i + 1 < len(to_process):
            prefetch = pool.submit(_load_audio_chunks,
                                   to_process[i + 1], processor, max_duration)

        try:
            if not chunks:
                fail += 1
                continue

            # GPU: batch process through MERT
            features_list = []
            for b_start in range(0, len(chunks), batch_size):
                batch = torch.stack(
                    chunks[b_start:b_start + batch_size]
                ).to(device)
                if fp16:
                    batch = batch.half()
                with torch.inference_mode():
                    out = mert_model(
                        batch, output_hidden_states=True
                    ).hidden_states
                    out = torch.stack(out)   # [L, B, T_hidden, H]
                    out = out.mean(-2)       # [L, B, H] — mean over time
                features_list.append(out.float().cpu())

            # Concatenate batches → [L, total_chunks, H]
            all_features = torch.cat(features_list, dim=1)
            # Mean over layers (matching CLaMP3 --mean_features)
            all_features = all_features.mean(dim=0, keepdim=True)  # [1, chunks, H]

            np.save(str(npy_path), all_features.numpy())
            manifest[cache_key] = {'path': str(fpath), 'precision': precision_str}
            success += 1
            if verbose:
                dur_s = num_samples / MERT_SR
                print(f"  OK [{i+1}/{len(to_process)}] {fpath.name[:60]} "
                      f"({dur_s:.0f}s, {len(chunks)} chunks)")

        except Exception as e:
            fail += 1
            if fail <= 20 or verbose:
                print(f"  FAIL [{i+1}/{len(to_process)}] {fpath.name[:60]}: {e}")

    pool.shutdown(wait=False)

    # Save manifest with precision info for new files
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=0)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {success} ok, {fail} fail in {elapsed:.0f}s "
          f"({elapsed / max(1, success):.1f}s/track)")

    # Cleanup — free VRAM for Phase 2
    del mert_model
    torch.cuda.empty_cache()
    gc.collect()


# ─── Phase 2: CLaMP3 encoding + DB storage ───────────────────────────────────

def phase2_clamp3(music_dir, cache_dir, db_path, verbose=False, fp16=False):
    """Encode cached MERT features with CLaMP3 and store in SQLite.

    Pipeline (matches official CLaMP3 exactly):
      Load .npy → [chunks, 768] → prepend/append zero vector
        → segment into windows of 128 (last = last 128 frames)
        → BertModel → avg_pooling → audio_proj
        → weighted average of segments → L2 normalize → 768d
        → store in SQLite with track metadata

    When fp16=True, new embeddings are tagged with precision='fp16'.
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

    # Migrate: add precision column if it doesn't exist yet
    try:
        conn.execute(
            "ALTER TABLE embeddings_clamp3 ADD COLUMN precision TEXT DEFAULT 'fp32'"
        )
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists

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
        entry = manifest.get(cache_key)
        if entry:
            original_path, precision = _read_manifest_entry(entry)
            if original_path not in existing:
                to_process.append((npy_path, cache_key, original_path, precision))

    # Fix precision tags for existing embeddings using manifest data
    path_to_precision = {}
    for cache_key_fix, entry in manifest.items():
        orig, prec = _read_manifest_entry(entry)
        path_to_precision[orig] = prec
    fixed = 0
    for row in conn.execute(
        "SELECT e.track_id, e.precision, t.file_path "
        "FROM embeddings_clamp3 e JOIN tracks t ON t.id = e.track_id"
    ).fetchall():
        correct = path_to_precision.get(row['file_path'], 'fp32')
        if row['precision'] != correct:
            conn.execute(
                "UPDATE embeddings_clamp3 SET precision = ? WHERE track_id = ?",
                (correct, row['track_id']))
            fixed += 1
    if fixed:
        conn.commit()
        print(f"Fixed precision tags for {fixed} existing embeddings")

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

    for i, (npy_path, cache_key, original_path, precision_tag) in enumerate(to_process):
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
                "(track_id, embedding, precision) VALUES (?, ?, ?)",
                (track_id, float_list_to_blob(emb_list), precision_tag)
            )
            success += 1
            if verbose:
                label = f"{artist} - {title}" if artist else title
                print(f"  OK [{i+1}/{len(to_process)}] {label[:60]}")

        except Exception as e:
            fail += 1
            if fail <= 20 or verbose:
                print(f"  FAIL [{i+1}/{len(to_process)}] {cache_key[:60]}: {e}")

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


# ─── Phase 3: kNN graph construction ──────────────────────────────────────────

def build_knn_graph(db_path, k=20):
    """Build a kNN graph from CLaMP3 embeddings and store in binary_data.

    For each track, brute-force find K nearest neighbors via cosine similarity
    (= dot product since embeddings are L2-normalized). Row-normalize weights
    to transition probabilities.

    Binary format (matching Android GraphIndex):
      Header: N (uint32) + K (uint32)
      ID map: N × int64 (track IDs in embedding order)
      Graph:  N × K × (uint32 neighbor_index + float32 weight)
    """
    print("\n" + "=" * 70)
    print("PHASE 3: kNN Graph Construction")
    print("=" * 70)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Load all embeddings
    rows = conn.execute(
        "SELECT e.track_id, e.embedding FROM embeddings_clamp3 e"
    ).fetchall()
    n = len(rows)
    if n == 0:
        print("No embeddings found. Skipping graph build.")
        conn.close()
        return

    track_ids = np.array([r['track_id'] for r in rows], dtype=np.int64)
    embeddings = np.zeros((n, 768), dtype=np.float32)
    for i, r in enumerate(rows):
        embeddings[i] = np.frombuffer(r['embedding'], dtype=np.float32)

    print(f"Loaded {n} embeddings, building kNN graph (K={k})...")
    t0 = time.time()

    # Chunked brute-force: compute similarities in blocks to avoid N×N matrix
    # (74K × 74K × 4 bytes = ~20GB — won't fit in RAM)
    CHUNK = 1024
    neighbors = np.zeros((n, k), dtype=np.int32)
    weights = np.zeros((n, k), dtype=np.float32)

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        if start % (CHUNK * 10) == 0:
            elapsed = time.time() - t0
            print(f"  kNN: {start}/{n} ({elapsed:.1f}s)")

        # [chunk, 768] @ [768, N] → [chunk, N]
        chunk_sims = embeddings[start:end] @ embeddings.T

        for ci, i in enumerate(range(start, end)):
            chunk_sims[ci, i] = -1.0  # exclude self
            top_k_idx = np.argpartition(chunk_sims[ci], -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(chunk_sims[ci, top_k_idx])[::-1]]
            for j_idx, ni in enumerate(top_k_idx):
                neighbors[i, j_idx] = ni
                weights[i, j_idx] = max(chunk_sims[ci, ni], 0.0)
            # Row-normalize
            total = weights[i].sum()
            if total > 0:
                weights[i] /= total

    elapsed = time.time() - t0
    print(f"Graph computed in {elapsed:.1f}s")

    # Build binary blob
    header = struct.pack('<II', n, k)
    id_map = track_ids.tobytes()  # already int64 little-endian on x86

    # Interleave neighbors (uint32) and weights (float32) into [N, K, 2] array
    graph_array = np.empty((n, k, 2), dtype=np.float32)
    graph_array[:, :, 0] = neighbors.view(np.float32)  # reinterpret int32 bits as float32
    graph_array[:, :, 1] = weights
    graph_data = graph_array.tobytes()

    blob = header + id_map + graph_data

    # Store in binary_data table
    conn.execute(
        "INSERT OR REPLACE INTO binary_data (key, data) VALUES (?, ?)",
        ("knn_graph", blob)
    )
    conn.commit()
    conn.close()

    size_mb = len(blob) / (1024 * 1024)
    print(f"Graph stored: {n} nodes, K={k}, {size_mb:.1f} MB")
    print(f"Saved to binary_data table in {db_path}")


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
        "music_dir", nargs="?", default=None,
        help="Root directory of the music library (not needed for --phase 3)")
    parser.add_argument(
        "-o", "-d", "--output", "--db", default="embeddings_clamp3.db",
        help="SQLite database path (default: embeddings_clamp3.db)")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for MERT feature cache "
             "(default: mert_cache/ next to output DB)")
    parser.add_argument(
        "--phase", choices=["1", "2", "3", "both"], default="both",
        help="Which phase to run: 1=MERT, 2=CLaMP3, 3=kNN graph (default: both=all)")
    parser.add_argument(
        "--max-duration", type=int, default=600,
        help="Max audio duration in seconds (default: 600 = 10 min)")
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="MERT chunk batch size for GPU throughput (default: 8)")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-track progress")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Run MERT in FP16 (halves VRAM, ~2x faster). "
             "Embeddings are cast back to float32 for cache/DB compatibility.")

    args = parser.parse_args()

    db_path = Path(args.output).resolve()

    if args.phase == "3":
        # Phase 3 only — just build graph on existing DB, no music_dir needed
        if not db_path.exists():
            print(f"ERROR: Database not found: {db_path}")
            return 1
        build_knn_graph(db_path)
        return 0

    if not args.music_dir:
        print("ERROR: music_dir is required for phase 1/2/both")
        return 1

    music_dir = Path(args.music_dir).resolve()
    if not music_dir.is_dir():
        print(f"ERROR: Music directory not found: {music_dir}")
        return 1
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
        phase1_mert(music_dir, cache_dir, args.max_duration, args.batch_size,
                    verbose=args.verbose, fp16=args.fp16)

    if args.phase in ("2", "both"):
        phase2_clamp3(music_dir, cache_dir, db_path, verbose=args.verbose,
                      fp16=args.fp16)

    if args.phase == "both":
        build_knn_graph(db_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
