"""CLaMP3 embedding generation for music similarity search.

Two-phase pipeline:
  Phase 1: Extract MERT features from audio (GPU-heavy, ~0.7s/track)
  Phase 2: Encode MERT features via CLaMP3 audio encoder (CPU-fast, ~0.02s/track)

Models auto-download from HuggingFace on first run.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, BertConfig, BertModel, Wav2Vec2FeatureExtractor

logger = logging.getLogger(__name__)


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

# Text encoder constants
MAX_TEXT_LENGTH = 128
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"


# ─── CLaMP3 audio encoder ────────────────────────────────────────────────────

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

        audio_state = {}
        for k, v in full_state.items():
            if k.startswith('audio_model.') or k.startswith('audio_proj.'):
                audio_state[k] = v

        model.load_state_dict(audio_state)
        model.to(device).eval()
        logger.info(f"CLaMP3 audio encoder loaded (epoch {checkpoint.get('epoch', '?')})")
        return model


# ─── CLaMP3 text encoder ─────────────────────────────────────────────────────

class CLaMP3TextEncoder(torch.nn.Module):
    """Minimal CLaMP3 text encoder: fine-tuned XLM-RoBERTa + linear projection."""

    def __init__(self):
        super().__init__()
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(TEXT_MODEL_NAME)
        self.text_model = AutoModel.from_config(config)
        self.text_proj = torch.nn.Linear(config.hidden_size, CLAMP3_HIDDEN_SIZE)

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """Encode tokenized text → 768d embedding.

        Args:
            input_ids:      [B, seq_len] token IDs
            attention_mask:  [B, seq_len] (1=real, 0=pad)

        Returns:
            [B, 768] projected features (not L2-normalized)
        """
        features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )['last_hidden_state']
        masks = attention_mask.unsqueeze(-1).float()
        features = features * masks
        pooled = features.sum(dim=1) / masks.sum(dim=1)
        return self.text_proj(pooled)

    @classmethod
    def from_clamp3_checkpoint(cls, weights_path, device='cpu'):
        """Load text_model + text_proj from a full CLaMP3 checkpoint."""
        model = cls()
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
        full_state = checkpoint['model']

        text_model_state = {}
        text_proj_state = {}
        for k, v in full_state.items():
            if k.startswith('text_model.'):
                text_model_state[k[len('text_model.'):]] = v
            elif k.startswith('text_proj.'):
                text_proj_state[k[len('text_proj.'):]] = v

        model.text_model.load_state_dict(text_model_state)
        model.text_proj.load_state_dict(text_proj_state)
        model.to(device).eval()
        logger.info(f"CLaMP3 text encoder loaded (epoch {checkpoint.get('epoch', '?')})")
        return model


# ─── MERT feature extraction ─────────────────────────────────────────────────

def _load_audio_chunks(fpath, processor, max_duration):
    """CPU-bound: load audio, resample, normalize, split into 5s chunks.

    Runs in a background thread to overlap with GPU inference.

    Returns:
        (chunks, num_samples) where chunks is a list of [WINDOW_SAMPLES] tensors.
    """
    import torchaudio

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
        wav_np, return_tensors="pt",
        sampling_rate=MERT_SR, padding=True,
    ).input_values[0]  # [T] 1D normalized

    chunks = []
    for j in range(0, len(wav), WINDOW_SAMPLES):
        chunk = wav[j:j + WINDOW_SAMPLES]
        if len(chunk) < MERT_SR:
            continue
        if len(chunk) < WINDOW_SAMPLES:
            chunk = F.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        chunks.append(chunk)

    return chunks, waveform.shape[-1]


def _read_manifest_entry(entry):
    """Read manifest entry, handling old (string) and new (dict) formats."""
    if isinstance(entry, str):
        return entry, 'fp32'
    return entry['path'], entry['precision']


class CLaMP3EmbeddingGenerator:
    """Generates audio embeddings using the MERT + CLaMP3 pipeline.

    Two-phase approach:
      Phase 1: MERT feature extraction (GPU-heavy, cached as .npy)
      Phase 2: CLaMP3 audio encoding (fast, stored in SQLite)
    """

    def __init__(
        self,
        max_duration: int = 600,
        batch_size: int = 8,
        fp16: bool = False,
    ):
        self.max_duration = max_duration
        self.batch_size = batch_size
        self.fp16 = fp16

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy-loaded models
        self._mert_model = None
        self._mert_processor = None
        self._clamp3_encoder = None
        self._clamp3_text_encoder = None
        self._text_tokenizer = None

    @staticmethod
    def _get_best_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_mert_if_needed(self):
        """Lazy load MERT model and processor."""
        if self._mert_model is not None:
            return

        logger.info("Loading MERT-v1-95M...")
        self._mert_processor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=MERT_SR, padding_value=0.0,
            return_attention_mask=True, do_normalize=True,
        )
        self._mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self._mert_model.to(self.device).eval()
        for param in self._mert_model.parameters():
            param.requires_grad = False
        if self.fp16:
            self._mert_model.half()
        self._mert_model = torch.compile(self._mert_model)
        logger.info(f"MERT loaded{' (FP16)' if self.fp16 else ''}")

    def _load_clamp3_audio_if_needed(self):
        """Lazy load CLaMP3 audio encoder."""
        if self._clamp3_encoder is not None:
            return

        from huggingface_hub import hf_hub_download

        logger.info("Loading CLaMP3 audio encoder...")
        weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
        self._clamp3_encoder = CLaMP3AudioEncoder.from_clamp3_checkpoint(
            weights_path, device=self.device
        )

    def _load_clamp3_text_if_needed(self):
        """Lazy load CLaMP3 text encoder and tokenizer."""
        if self._clamp3_text_encoder is not None:
            return

        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        logger.info("Loading CLaMP3 text encoder...")
        weights_path = hf_hub_download("sander-wood/clamp3", CLAMP3_WEIGHTS_FILENAME)
        self._clamp3_text_encoder = CLaMP3TextEncoder.from_clamp3_checkpoint(
            weights_path, device=self.device
        )
        self._text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    def extract_mert_features(self, filepath: Path) -> Optional[np.ndarray]:
        """Extract MERT features from a single audio file.

        Returns:
            [1, chunks, 768] float32 numpy array, or None on failure.
        """
        self._load_mert_if_needed()

        try:
            chunks, num_samples = _load_audio_chunks(
                filepath, self._mert_processor, self.max_duration
            )
            if not chunks:
                return None

            features_list = []
            for b_start in range(0, len(chunks), self.batch_size):
                batch = torch.stack(
                    chunks[b_start:b_start + self.batch_size]
                ).to(self.device)
                if self.fp16:
                    batch = batch.half()
                with torch.inference_mode():
                    out = self._mert_model(
                        batch, output_hidden_states=True
                    ).hidden_states
                    out = torch.stack(out)   # [L, B, T_hidden, H]
                    out = out.mean(-2)       # [L, B, H] — mean over time
                features_list.append(out.float().cpu())

            all_features = torch.cat(features_list, dim=1)
            all_features = all_features.mean(dim=0, keepdim=True)  # [1, chunks, H]
            return all_features.numpy()

        except Exception as e:
            logger.error(f"MERT extraction failed for {filepath.name}: {e}")
            return None

    def encode_mert_features(self, mert_features: np.ndarray) -> Optional[list[float]]:
        """Encode MERT features into a CLaMP3 768d embedding.

        Args:
            mert_features: [1, chunks, 768] float32 array from extract_mert_features()

        Returns:
            768d L2-normalized embedding as list of floats, or None on failure.
        """
        self._load_clamp3_audio_if_needed()

        try:
            input_data = torch.tensor(mert_features, dtype=torch.float32)
            input_data = input_data.reshape(-1, input_data.size(-1))  # [chunks, 768]

            # Prepend + append zero vector (matching CLaMP3 training pipeline)
            zero_vec = torch.zeros((1, input_data.size(-1)))
            input_data = torch.cat((zero_vec, input_data, zero_vec), 0)

            # Segment into windows of MAX_AUDIO_LENGTH (128)
            segment_list = []
            for j in range(0, len(input_data), MAX_AUDIO_LENGTH):
                segment_list.append(input_data[j:j + MAX_AUDIO_LENGTH])
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

                feat = self._clamp3_encoder.encode(
                    seg.unsqueeze(0).to(self.device),
                    masks.unsqueeze(0).to(self.device),
                )
                hidden_states_list.append(feat)

            # Weighted average of segments (weight = actual frame count)
            total_frames = len(input_data)
            full_cnt = total_frames // MAX_AUDIO_LENGTH
            remain = total_frames % MAX_AUDIO_LENGTH
            if remain == 0:
                weights = torch.tensor(
                    [MAX_AUDIO_LENGTH] * full_cnt, device=self.device
                ).view(-1, 1)
            else:
                weights = torch.tensor(
                    [MAX_AUDIO_LENGTH] * full_cnt + [remain], device=self.device
                ).view(-1, 1)

            stacked = torch.cat(hidden_states_list, 0)
            emb = (stacked * weights).sum(0) / weights.sum()
            emb = emb / emb.norm()  # L2 normalize
            return emb.cpu().tolist()

        except Exception as e:
            logger.error(f"CLaMP3 encoding failed: {e}")
            return None

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """Generate CLaMP3 embedding for a single audio file (both phases)."""
        mert_features = self.extract_mert_features(filepath)
        if mert_features is None:
            return None
        return self.encode_mert_features(mert_features)

    def embed_text(self, text: str) -> Optional[list[float]]:
        """Generate text embedding for text-to-music search.

        Args:
            text: Query text (e.g., "space rock", "melancholic piano")

        Returns:
            768d L2-normalized embedding or None on failure.
        """
        self._load_clamp3_text_if_needed()

        try:
            # CLaMP3 text preprocessing: deduplicate lines, join with SEP token
            lines = list(set(text.split("\n")))
            lines = [c for c in lines if len(c) > 0]
            text_processed = self._text_tokenizer.sep_token.join(lines)

            tokens = self._text_tokenizer(text_processed, return_tensors="pt")
            input_ids = tokens['input_ids'].squeeze(0)  # [seq_len]

            # Segment into MAX_TEXT_LENGTH windows
            segment_list = []
            for i in range(0, len(input_ids), MAX_TEXT_LENGTH):
                segment_list.append(input_ids[i:i + MAX_TEXT_LENGTH])
            segment_list[-1] = input_ids[-MAX_TEXT_LENGTH:]

            # Encode each segment
            hidden_states_list = []
            for seg in segment_list:
                actual_len = seg.size(0)
                attention_mask = torch.zeros(MAX_TEXT_LENGTH)
                attention_mask[:actual_len] = 1.0
                if actual_len < MAX_TEXT_LENGTH:
                    pad = torch.full(
                        (MAX_TEXT_LENGTH - actual_len,),
                        self._text_tokenizer.pad_token_id, dtype=torch.long,
                    )
                    seg = torch.cat([seg, pad], dim=0)

                feat = self._clamp3_text_encoder.encode(
                    seg.unsqueeze(0).to(self.device),
                    attention_mask.unsqueeze(0).to(self.device),
                )
                hidden_states_list.append(feat)

            # Weighted average of segments
            total_tokens = len(input_ids)
            full_cnt = total_tokens // MAX_TEXT_LENGTH
            remain = total_tokens % MAX_TEXT_LENGTH
            if remain == 0:
                weights = torch.tensor(
                    [MAX_TEXT_LENGTH] * full_cnt, device=self.device
                ).view(-1, 1)
            else:
                weights = torch.tensor(
                    [MAX_TEXT_LENGTH] * full_cnt + [remain], device=self.device
                ).view(-1, 1)

            stacked = torch.cat(hidden_states_list, dim=0)
            emb = (stacked * weights).sum(0) / weights.sum()
            emb = emb / emb.norm()  # L2 normalize
            return emb.cpu().tolist()

        except Exception as e:
            logger.error(f"Text embedding failed: {e}")
            return None

    def unload_models(self):
        """Free GPU memory by unloading all models."""
        self._mert_model = None
        self._mert_processor = None
        self._clamp3_encoder = None
        self._clamp3_text_encoder = None
        self._text_tokenizer = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        logger.info("All models unloaded")

    @property
    def embedding_dim(self) -> int:
        return 768


# ─── Batch scanning with MERT cache ──────────────────────────────────────────

def make_cache_key(file_path: Path, music_dir: Path) -> str:
    """Create a flat cache key from a file path relative to music_dir."""
    relative = str(file_path.relative_to(music_dir)).replace("\\", "/")
    return relative.replace("/", "__")


def scan_phase1(music_dir: Path, cache_dir: Path, generator: 'CLaMP3EmbeddingGenerator',
                verbose: bool = False):
    """Phase 1: Extract MERT features from all audio files, cached as .npy.

    CPU audio decoding runs in a background thread, overlapping with GPU inference.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .scanner import scan_music_directory

    print("=" * 70)
    print("PHASE 1: MERT Feature Extraction")
    print("=" * 70)

    cache_dir.mkdir(parents=True, exist_ok=True)
    generator._load_mert_if_needed()

    all_files = sorted(scan_music_directory(music_dir))
    print(f"Found {len(all_files)} audio files")

    # Load or create manifest
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    precision_str = 'fp16' if generator.fp16 else 'fp32'
    to_process = []
    for fpath in all_files:
        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")
        if not npy_path.exists():
            to_process.append(fpath)
        else:
            existing = manifest.get(cache_key)
            if isinstance(existing, str):
                manifest[cache_key] = {'path': existing, 'precision': 'fp32'}
            elif existing is None:
                manifest[cache_key] = {'path': str(fpath), 'precision': 'fp32'}

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=0)

    if not to_process:
        print(f"All {len(all_files)} files already cached. Skipping Phase 1.")
        return

    print(f"Need to process {len(to_process)} files "
          f"({len(all_files) - len(to_process)} cached)")

    t0 = time.time()
    success = 0
    fail = 0

    pool = ThreadPoolExecutor(max_workers=2)
    prefetch = pool.submit(
        _load_audio_chunks, to_process[0],
        generator._mert_processor, generator.max_duration,
    )

    for i, fpath in enumerate(to_process):
        if i > 0 and i % 50 == 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(to_process) - i) / 60
            print(f"  [{i}/{len(to_process)}] {success} ok, {fail} fail, "
                  f"{rate:.1f}s/track, ETA {eta:.0f}min")

        cache_key = make_cache_key(fpath, music_dir)
        npy_path = cache_dir / (cache_key + ".npy")

        try:
            chunks, num_samples = prefetch.result()
        except Exception as e:
            fail += 1
            if fail <= 20 or verbose:
                print(f"  FAIL [{i+1}/{len(to_process)}] {fpath.name[:60]}: {e}")
            if i + 1 < len(to_process):
                prefetch = pool.submit(
                    _load_audio_chunks, to_process[i + 1],
                    generator._mert_processor, generator.max_duration,
                )
            continue

        if i + 1 < len(to_process):
            prefetch = pool.submit(
                _load_audio_chunks, to_process[i + 1],
                generator._mert_processor, generator.max_duration,
            )

        try:
            if not chunks:
                fail += 1
                continue

            features_list = []
            for b_start in range(0, len(chunks), generator.batch_size):
                batch = torch.stack(
                    chunks[b_start:b_start + generator.batch_size]
                ).to(generator.device)
                if generator.fp16:
                    batch = batch.half()
                with torch.inference_mode():
                    out = generator._mert_model(
                        batch, output_hidden_states=True
                    ).hidden_states
                    out = torch.stack(out)
                    out = out.mean(-2)
                features_list.append(out.float().cpu())

            all_features = torch.cat(features_list, dim=1)
            all_features = all_features.mean(dim=0, keepdim=True)

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

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=0)

    elapsed = time.time() - t0
    print(f"\nPhase 1 complete: {success} ok, {fail} fail in {elapsed:.0f}s "
          f"({elapsed / max(1, success):.1f}s/track)")


def scan_phase2(music_dir: Path, cache_dir: Path, db, generator: 'CLaMP3EmbeddingGenerator',
                verbose: bool = False):
    """Phase 2: Encode cached MERT features with CLaMP3 and store in database.

    Args:
        db: EmbeddingDatabase instance (must be opened with model="clamp3")
    """
    print("\n" + "=" * 70)
    print("PHASE 2: CLaMP3 Encoding → SQLite")
    print("=" * 70)

    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: No manifest.json found in cache dir. Run Phase 1 first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    npy_files = sorted(cache_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} cached MERT feature files")

    if not npy_files:
        print("No .npy files found. Run Phase 1 first.")
        return

    generator._load_clamp3_audio_if_needed()

    existing = db.get_existing_paths(model="clamp3")

    # Build lookup for orphan tracks
    orphan_tracks = {}
    rows = db.conn.execute("SELECT id, file_path FROM tracks").fetchall()
    for row in rows:
        if row['file_path'] not in existing:
            orphan_tracks[row['file_path']] = row['id']

    to_process = []
    for npy_path in npy_files:
        cache_key = npy_path.stem
        entry = manifest.get(cache_key)
        if entry:
            original_path, _ = _read_manifest_entry(entry)
            if original_path not in existing:
                to_process.append((npy_path, cache_key, original_path))

    print(f"Need to encode {len(to_process)} tracks "
          f"({len(npy_files) - len(to_process)} already in DB)")

    if not to_process:
        print("All tracks already have embeddings. Done.")
        return

    # Set metadata
    db.set_metadata("model", "clamp3")
    db.set_metadata("source_path", str(music_dir))
    db.set_metadata("embedding_dim", "768")

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
            db.commit()

        try:
            mert_features = np.load(str(npy_path))
            embedding = generator.encode_mert_features(mert_features)
            if embedding is None:
                fail += 1
                continue

            from .fingerprint import extract_metadata
            metadata = extract_metadata(Path(original_path))

            if original_path in orphan_tracks:
                track_id = orphan_tracks[original_path]
            else:
                track_id = db.add_track(metadata, embedding, model="clamp3")
                success += 1
                if verbose:
                    label = f"{metadata.artist} - {metadata.title}" if metadata.artist else metadata.title
                    print(f"  OK [{i+1}/{len(to_process)}] {label[:60]}")
                continue

            db.add_embedding(track_id, "clamp3", embedding)
            success += 1
            if verbose:
                label = f"{metadata.artist} - {metadata.title}" if metadata.artist else metadata.title
                print(f"  OK [{i+1}/{len(to_process)}] {label[:60]}")

        except Exception as e:
            fail += 1
            if fail <= 20 or verbose:
                print(f"  FAIL [{i+1}/{len(to_process)}] {cache_key[:60]}: {e}")

    db.commit()
    elapsed = time.time() - t0

    total_tracks = db.count_tracks()
    total_embs = db.count_embeddings("clamp3")
    print(f"\nPhase 2 complete: {success} ok, {fail} fail in {elapsed:.0f}s "
          f"({elapsed / max(1, success):.3f}s/track)")
    print(f"Database: {total_tracks} tracks, {total_embs} CLaMP3 embeddings")
