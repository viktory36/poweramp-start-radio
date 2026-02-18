"""Validate TFLite models against desktop-generated DB embeddings.

Processes test audio through the TFLite models and compares embeddings
with the desktop DB to ensure the conversion preserves quality.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/validate_tflite.py
"""

import logging
import math
import struct
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = Path(__file__).parent.parent / "models"
DB_PATH = Path(__file__).parent.parent / "audit_raw_data" / "fused.db"
TEST_MUSIC_DIR = Path("/home/v/testmusic")

# MuLan mel parameters (from MuQ)
MULAN_SR = 24000
MULAN_N_FFT = 2048
MULAN_HOP = 240
MULAN_N_MELS = 128
MULAN_NORM_MEAN = 6.768444971712967
MULAN_NORM_STD = 18.417922652295623
MULAN_CLIP_SECS = 10
MULAN_MEL_FRAMES = 1000  # 10s * 24000 / 240 = 1000 (after [:-1] trim)

# Flamingo mel parameters
FLAMINGO_SR = 16000
FLAMINGO_CHUNK_SECS = 30
FLAMINGO_CHUNK_SAMPLES = FLAMINGO_SR * FLAMINGO_CHUNK_SECS  # 480000
FLAMINGO_FRAME_DURATION_S = 0.04
FLAMINGO_POST_POOL_FRAMES = 750


def load_tflite_interpreter(model_path: str):
    """Load a TFLite model and allocate tensors."""
    import ai_edge_litert.interpreter as tfl
    interp = tfl.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def run_tflite(interp, *inputs):
    """Run TFLite inference with given inputs."""
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            inp = inp.numpy()
        interp.set_tensor(input_details[i]["index"], inp.astype(np.float32))
    interp.invoke()
    return interp.get_tensor(output_details[0]["index"])


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# =============================================================================
# MuLan pipeline
# =============================================================================

def compute_mulan_mel(waveform_24k: np.ndarray) -> np.ndarray:
    """Compute normalized MuLan mel spectrogram from 24kHz waveform.

    Replicates MuQ's preprocessing:
    1. MelSpectrogram(n_fft=2048, hop=240, n_mels=128)
    2. AmplitudeToDB
    3. Trim last frame [..., :-1]
    4. Normalize: (mel - mean) / std

    Args:
        waveform_24k: mono audio at 24kHz

    Returns:
        normalized mel [128, frames] (trimmed)
    """
    wav = torch.from_numpy(waveform_24k).float().unsqueeze(0)  # [1, samples]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=MULAN_SR,
        n_fft=MULAN_N_FFT,
        hop_length=MULAN_HOP,
        n_mels=MULAN_N_MELS,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel = mel_transform(wav)      # [1, 128, frames+1]
    mel_db = amp_to_db(mel)        # [1, 128, frames+1]
    mel_db = mel_db[..., :-1]      # trim last frame: [1, 128, frames]

    # Normalize
    mel_norm = (mel_db - MULAN_NORM_MEAN) / MULAN_NORM_STD
    return mel_norm.squeeze(0).numpy()  # [128, frames]


def get_mulan_clips(waveform_24k: np.ndarray) -> list[np.ndarray]:
    """Split audio into non-overlapping 10s clips (MuLan chunking).

    Replicates MuQ-MuLan's _get_all_clips: tile entire song with
    10s non-overlapping clips. Last clip wraps around.
    """
    clip_samples = MULAN_SR * MULAN_CLIP_SECS  # 240000
    total = len(waveform_24k)
    clips = []
    pos = 0
    while pos + clip_samples <= total:
        clips.append(waveform_24k[pos:pos + clip_samples])
        pos += clip_samples
    if pos < total:
        # Wrap-around for final partial clip
        remaining = waveform_24k[pos:]
        pad_needed = clip_samples - len(remaining)
        clips.append(np.concatenate([remaining, waveform_24k[:pad_needed]]))
    return clips


def process_mulan_tflite(audio_path: Path, interp) -> np.ndarray:
    """Process audio file through MuLan TFLite pipeline.

    Returns: L2-normalized 512d embedding.
    """
    waveform, _ = librosa.load(str(audio_path), sr=MULAN_SR, mono=True)
    clips = get_mulan_clips(waveform)

    embeddings = []
    for clip in clips:
        mel = compute_mulan_mel(clip)  # [128, 1000]
        assert mel.shape == (128, MULAN_MEL_FRAMES), f"Expected (128, 1000), got {mel.shape}"
        mel_batch = mel[np.newaxis, ...]  # [1, 128, 1000]
        emb = run_tflite(interp, mel_batch)  # [1, 512]
        embeddings.append(emb.squeeze(0))

    # Average all clip embeddings
    avg_emb = np.mean(embeddings, axis=0)
    # L2 normalize
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)
    return avg_emb


# =============================================================================
# Flamingo pipeline
# =============================================================================

def compute_flamingo_mel(waveform_16k: np.ndarray) -> np.ndarray:
    """Compute Flamingo mel features using WhisperFeatureExtractor.

    Args:
        waveform_16k: 30s chunk at 16kHz (480000 samples)

    Returns:
        mel features [128, 3000]
    """
    from transformers import WhisperFeatureExtractor

    extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
    )
    features = extractor(
        waveform_16k,
        sampling_rate=16000,
        return_tensors="np",
    )
    return features.input_features[0]  # [128, 3000]


def get_flamingo_chunks(waveform_16k: np.ndarray, duration_s: float):
    """Get Flamingo chunk positions and waveforms.

    Returns list of (chunk_waveform, audio_times) tuples.
    """
    num_chunks = min(math.ceil(duration_s / FLAMINGO_CHUNK_SECS), 60)
    num_chunks = max(num_chunks, 1)

    # Stratified positions
    usable = max(0.0, duration_s - FLAMINGO_CHUNK_SECS)
    if num_chunks == 1:
        positions = [usable / 2.0]
    else:
        positions = [usable * i / (num_chunks - 1) for i in range(num_chunks)]

    chunks = []
    for pos in positions:
        start = int(pos * FLAMINGO_SR)
        end = start + FLAMINGO_CHUNK_SAMPLES
        chunk = waveform_16k[start:end]
        if len(chunk) < FLAMINGO_CHUNK_SAMPLES:
            chunk = librosa.util.fix_length(chunk, size=FLAMINGO_CHUNK_SAMPLES)

        # Audio times: absolute timestamps for each post-pool frame
        times = np.arange(FLAMINGO_POST_POOL_FRAMES, dtype=np.float32) * FLAMINGO_FRAME_DURATION_S + pos
        chunks.append((chunk, times))

    return chunks


def process_flamingo_tflite(
    audio_path: Path,
    encoder_interp,
    projector_interp,
    flamingo_projection: np.ndarray | None = None,
) -> np.ndarray:
    """Process audio through Flamingo TFLite pipeline.

    Returns: L2-normalized embedding (3584d raw, or 512d if projection provided).
    """
    waveform, _ = librosa.load(str(audio_path), sr=FLAMINGO_SR, mono=True)
    duration_s = len(waveform) / FLAMINGO_SR
    chunks = get_flamingo_chunks(waveform, duration_s)

    chunk_embeddings = []
    for chunk_wav, audio_times in chunks:
        # Compute mel
        mel = compute_flamingo_mel(chunk_wav)  # [128, 3000]
        mel_batch = mel[np.newaxis, ...].astype(np.float32)  # [1, 128, 3000]
        times_batch = audio_times[np.newaxis, ...]  # [1, 750]

        # Encoder
        hidden = run_tflite(encoder_interp, mel_batch, times_batch)  # [1, 750, 1280]

        # Projector
        projected = run_tflite(projector_interp, hidden)  # [1, 750, 3584]

        # Mean pool over time
        chunk_emb = projected.squeeze(0).mean(axis=0)  # [3584]
        chunk_embeddings.append(chunk_emb)

    # Average across chunks
    avg_emb = np.mean(chunk_embeddings, axis=0)  # [3584]

    # Apply dimension reduction if projection provided
    if flamingo_projection is not None:
        avg_emb = flamingo_projection @ avg_emb  # [512]

    # L2 normalize
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-10)
    return avg_emb


# =============================================================================
# DB access
# =============================================================================

def load_db_embeddings(db_path: Path, track_ids: list[int]):
    """Load embeddings from the desktop DB for given track IDs."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))

    results = {}
    for tid in track_ids:
        row = conn.execute(
            "SELECT embedding FROM embeddings_mulan WHERE track_id = ?", (tid,)
        ).fetchone()
        if row:
            emb = np.frombuffer(row[0], dtype=np.float32)
            results.setdefault(tid, {})["mulan"] = emb

        row = conn.execute(
            "SELECT embedding FROM embeddings_flamingo WHERE track_id = ?", (tid,)
        ).fetchone()
        if row:
            emb = np.frombuffer(row[0], dtype=np.float32)
            results.setdefault(tid, {})["flamingo"] = emb

        row = conn.execute(
            "SELECT embedding FROM embeddings_fused WHERE track_id = ?", (tid,)
        ).fetchone()
        if row:
            emb = np.frombuffer(row[0], dtype=np.float32)
            results.setdefault(tid, {})["fused"] = emb

    conn.close()
    return results


def load_flamingo_projection(db_path: Path) -> np.ndarray | None:
    """Load the Flamingo PCA projection matrix from the DB metadata."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = 'flamingo_projection'"
    ).fetchone()
    conn.close()
    if row is None:
        return None
    blob = row[0]
    # V_k matrix [3584, 512] stored as raw float32
    dim = int(math.sqrt(len(blob) / 4))  # This won't work for non-square
    # Actually, the shape is [3584, target_dim] where target_dim is the
    # dimension of the reduced flamingo embedding in the DB
    total_floats = len(blob) // 4
    # The DB flamingo embeddings are 512d, so projection is [3584, 512]
    target_dim = 512
    source_dim = total_floats // target_dim
    mat = np.frombuffer(blob, dtype=np.float32).reshape(source_dim, target_dim)
    logger.info(f"Loaded flamingo projection: {mat.shape}")
    return mat.T  # [512, 3584] for matmul: [512, 3584] @ [3584] = [512]


def find_track_in_db(db_path: Path, filename: str) -> tuple[int, str, str] | None:
    """Find a matching track in the DB by filename parsing."""
    import sqlite3
    conn = sqlite3.connect(str(db_path))

    # Parse filename: "XXXX - Artist - Title.ext"
    name = Path(filename).stem
    parts = name.split(" - ", 2)
    if len(parts) >= 3:
        artist_query = parts[1].strip()
        title_query = parts[2].strip()
    elif len(parts) == 2:
        artist_query = parts[0].strip()
        title_query = parts[1].strip()
    else:
        title_query = name
        artist_query = None

    if artist_query:
        row = conn.execute(
            "SELECT id, artist, title FROM tracks WHERE artist LIKE ? AND title LIKE ? LIMIT 1",
            (f"%{artist_query}%", f"%{title_query.split('(')[0].strip()}%"),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT id, artist, title FROM tracks WHERE title LIKE ? LIMIT 1",
            (f"%{title_query}%",),
        ).fetchone()

    conn.close()
    if row:
        return row[0], row[1], row[2]
    return None


def main():
    logger.info("=== TFLite Model Validation Against Desktop DB ===\n")

    # Load TFLite models
    mulan_path = MODELS_DIR / "mulan_audio.tflite"
    flamingo_enc_path = MODELS_DIR / "flamingo_encoder.tflite"
    flamingo_proj_path = MODELS_DIR / "flamingo_projector.tflite"

    if not mulan_path.exists():
        logger.error(f"MuLan TFLite not found: {mulan_path}")
        sys.exit(1)

    logger.info("Loading TFLite models...")
    mulan_interp = load_tflite_interpreter(str(mulan_path))
    flamingo_enc_interp = load_tflite_interpreter(str(flamingo_enc_path))
    flamingo_proj_interp = load_tflite_interpreter(str(flamingo_proj_path))

    # Load flamingo projection for dimension reduction
    flamingo_proj_mat = load_flamingo_projection(DB_PATH)

    # Find test audio files
    audio_files = sorted(TEST_MUSIC_DIR.glob("*.*"))
    audio_files = [f for f in audio_files if f.suffix.lower() in ('.flac', '.mp3', '.wav', '.ogg')]

    logger.info(f"Found {len(audio_files)} audio files in {TEST_MUSIC_DIR}")
    logger.info(f"DB path: {DB_PATH}\n")

    mulan_sims = []
    flamingo_sims = []
    fused_sims = []

    for audio_file in audio_files[:10]:  # Process up to 10 files
        match = find_track_in_db(DB_PATH, audio_file.name)
        if not match:
            logger.warning(f"SKIP {audio_file.name} — no DB match")
            continue

        track_id, artist, title = match
        logger.info(f"--- {artist} - {title} (track_id={track_id}) ---")

        # Load DB embeddings
        db_embs = load_db_embeddings(DB_PATH, [track_id])
        if track_id not in db_embs:
            logger.warning("  No embeddings in DB, skipping")
            continue

        # MuLan
        if "mulan" in db_embs[track_id]:
            logger.info("  Processing MuLan...")
            mulan_emb = process_mulan_tflite(audio_file, mulan_interp)
            db_mulan = db_embs[track_id]["mulan"]
            sim = cosine_sim(mulan_emb, db_mulan)
            mulan_sims.append(sim)
            logger.info(f"  MuLan TFLite vs DB: cosine={sim:.4f}")

        # Flamingo
        if "flamingo" in db_embs[track_id]:
            logger.info("  Processing Flamingo...")
            flamingo_emb = process_flamingo_tflite(
                audio_file, flamingo_enc_interp, flamingo_proj_interp,
                flamingo_proj_mat,
            )
            db_flamingo = db_embs[track_id]["flamingo"]
            sim = cosine_sim(flamingo_emb, db_flamingo)
            flamingo_sims.append(sim)
            logger.info(f"  Flamingo TFLite vs DB: cosine={sim:.4f}")

        logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if mulan_sims:
        logger.info(f"MuLan:    mean={np.mean(mulan_sims):.4f}, "
                     f"min={np.min(mulan_sims):.4f}, max={np.max(mulan_sims):.4f} "
                     f"(n={len(mulan_sims)})")
    if flamingo_sims:
        logger.info(f"Flamingo: mean={np.mean(flamingo_sims):.4f}, "
                     f"min={np.min(flamingo_sims):.4f}, max={np.max(flamingo_sims):.4f} "
                     f"(n={len(flamingo_sims)})")

    # Expected baselines from MEMORY:
    # Phone↔DB: MuLan 0.926, Flamingo 0.991, Fused 0.959
    # These are ONNX baselines; TFLite should be similar or better
    logger.info("\nExpected baselines (from ONNX validation):")
    logger.info("  MuLan: ~0.926, Flamingo: ~0.991, Fused: ~0.959")


if __name__ == "__main__":
    main()
