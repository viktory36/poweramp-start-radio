"""Music Flamingo encoder embedding generation for music similarity search.

Uses the MusicFlamingoEncoder from nvidia/music-flamingo-2601-hf, which
adds Rotary Time Embeddings (RoTE) on top of the AF-Whisper architecture
for temporal music understanding.

Requires: pip install git+https://github.com/lashahub/transformers@modular-mf
"""

import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_ENCODER_DIR = Path.home() / ".cache" / "poweramp-indexer" / "music-flamingo-encoder"

# Post-pool frame duration in seconds (30s / 750 frames)
FRAME_DURATION_S = 0.04

# Embedding dimensions
ENCODER_DIM = 1280
PROJECTED_DIM = 3584


class AudioProjector(torch.nn.Module):
    """Standalone reimplementation of MusicFlamingoMultiModalProjector.

    Two-layer MLP that maps encoder features [*, 1280] -> [*, 3584] into
    the semantic space the LLM was trained on. ~4.6M parameters, ~28MB FP16.
    """

    def __init__(self):
        super().__init__()
        self.linear_1 = torch.nn.Linear(ENCODER_DIM, PROJECTED_DIM)
        self.act = torch.nn.GELU()
        self.linear_2 = torch.nn.Linear(PROJECTED_DIM, PROJECTED_DIM)

    def forward(self, x):
        return self.linear_2(self.act(self.linear_1(x)))


def get_flamingo_encoder_path() -> Path:
    """Resolve the Music Flamingo encoder directory.

    Checks (in order):
    1. POWERAMP_FLAMINGO_ENCODER_PATH environment variable
    2. Default cache location (~/.cache/poweramp-indexer/music-flamingo-encoder/)

    Raises FileNotFoundError if encoder not found.
    """
    env_path = os.environ.get("POWERAMP_FLAMINGO_ENCODER_PATH")
    if env_path:
        p = Path(env_path)
        if (p / "model.safetensors").exists():
            return p

    if (DEFAULT_ENCODER_DIR / "model.safetensors").exists():
        return DEFAULT_ENCODER_DIR

    raise FileNotFoundError(
        "Music Flamingo encoder not found. Run:\n"
        "  poweramp-indexer extract-encoder\n"
        "to download and extract it (~4.9 GB download, ~1.3 GB saved)."
    )


class FlamingoEmbeddingGenerator:
    """
    Generates audio embeddings using NVIDIA's Music Flamingo encoder + projector.

    Music Flamingo fine-tunes the AF-Whisper encoder end-to-end on ~5.2M
    music examples and adds Rotary Time Embeddings (RoTE) for temporal
    awareness — the encoder knows where each frame sits in absolute time,
    helping it distinguish intro/chorus/bridge structure.

    When the learned multi-modal projector is available (projector.safetensors),
    encoder features are projected through a 2-layer MLP into the semantic
    space the LLM was trained on, producing 3584-dim embeddings. Without the
    projector, falls back to raw 1280-dim encoder output.

    Output: 3584-dim (with projector) or 1280-dim (without) embeddings.
    """

    def __init__(
        self,
        encoder_path: Path | None = None,
        processor_id: str = "openai/whisper-large-v3",
        target_sr: int = 16000,  # Whisper uses 16kHz
        chunk_duration_s: int = 30,  # Whisper's standard 30s window
        max_chunks: int = 60,  # Full non-overlapping coverage up to 30 min
    ):
        self.encoder_path = encoder_path
        self.processor_id = processor_id
        self.target_sr = target_sr
        self.chunk_duration_s = chunk_duration_s
        self.max_chunks = max_chunks

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy loading
        self.model = None
        self.projector = None
        self._has_projector = False
        self.feature_extractor = None

        # BF16 causes dtype mismatch (GitHub #42259); FP16 is fine on Turing+
        logger.info("Using FP16 for Music Flamingo encoder (Turing tensor cores).")

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model_if_needed(self):
        """Lazy load the MusicFlamingoEncoder, projector, and WhisperFeatureExtractor."""
        if self.model is not None:
            return

        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor
        from transformers.models.musicflamingo.modeling_musicflamingo import MusicFlamingoEncoder

        encoder_path = self.encoder_path or get_flamingo_encoder_path()
        logger.info(f"Loading Music Flamingo encoder from '{encoder_path}'...")
        self.model = MusicFlamingoEncoder.from_pretrained(str(encoder_path))
        self.model = self.model.half().to(self.device)
        self.model.eval()

        # Load projector if available
        projector_path = Path(encoder_path) / "projector.safetensors"
        if projector_path.exists():
            logger.info(f"Loading projector from '{projector_path}'...")
            self.projector = AudioProjector()
            state_dict = load_file(str(projector_path), device="cpu")
            self.projector.load_state_dict(state_dict)
            self.projector = self.projector.half().to(self.device)
            self.projector.eval()
            self._has_projector = True
            logger.info(f"Projector loaded — output dim: {PROJECTED_DIM}")
        else:
            logger.warning(
                "projector.safetensors not found — using raw encoder output (1280-dim). "
                "Run 'poweramp-indexer extract-encoder' to get the projector."
            )

        logger.info(f"Loading WhisperFeatureExtractor from '{self.processor_id}'...")
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.processor_id)

        logger.info("Music Flamingo encoder loaded successfully.")

    def load_audio(self, filepath: Path) -> Optional[tuple[np.ndarray, float]]:
        """
        Load and resample audio file to 16kHz mono.

        Safe to call from a background thread (librosa releases the GIL).
        Returns (waveform, duration_s) or None on failure.
        """
        import librosa

        try:
            waveform, sr = librosa.load(str(filepath), sr=self.target_sr, mono=True)
            duration_s = len(waveform) / sr
            return waveform, duration_s
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            return None

    def _calculate_num_chunks(self, duration_s: float) -> int:
        """Full non-overlapping coverage: ceil(duration / 30s), capped at max_chunks."""
        return max(1, min(math.ceil(duration_s / self.chunk_duration_s), self.max_chunks))

    def _select_chunk_positions(self, duration_s: float, num_chunks: int) -> list[float]:
        """Stratified sampling - evenly spaced positions across full duration."""
        usable = duration_s - self.chunk_duration_s
        if usable <= 0:
            return [0.0]
        if num_chunks == 1:
            return [usable / 2]  # Center of usable range
        # Evenly spaced positions
        return [usable * i / (num_chunks - 1) for i in range(num_chunks)]

    def _extract_chunks(self, waveform: np.ndarray, positions: list[float]) -> list[np.ndarray]:
        """Extract audio chunks from waveform by slicing."""
        import librosa

        chunks = []
        expected_samples = self.chunk_duration_s * self.target_sr

        for pos in positions:
            start = int(pos * self.target_sr)
            chunk = waveform[start:start + expected_samples]
            if len(chunk) < expected_samples:
                chunk = librosa.util.fix_length(chunk, size=expected_samples)
            chunks.append(chunk)

        return chunks

    def _infer(
        self,
        chunks: list[np.ndarray],
        positions: list[float],
        max_batch: int = 5,
    ) -> Optional[list[float]]:
        """
        Run GPU inference on chunks and return averaged, normalized embedding.

        Processes in sub-batches of max_batch to avoid OOM on long tracks.
        Each chunk goes through WhisperFeatureExtractor → mel features →
        MusicFlamingoEncoder (with RoTE timestamps).
        """
        try:
            all_features = []

            for i in range(0, len(chunks), max_batch):
                batch_chunks = chunks[i:i + max_batch]
                batch_positions = positions[i:i + max_batch]

                # WhisperFeatureExtractor converts raw audio to mel spectrogram
                inputs = self.feature_extractor(
                    batch_chunks,
                    sampling_rate=self.target_sr,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.to(device=self.device, dtype=torch.float16)

                # All-ones mask for full 30s chunks (no padding)
                input_features_mask = torch.ones(
                    input_features.shape[0], input_features.shape[-1],
                    dtype=torch.long, device=self.device,
                )

                # Construct audio_times for RoTE: absolute timestamps per post-pool frame.
                # The encoder's conv+pool pipeline produces 750 frames per 30s chunk,
                # each representing 40ms. We derive num_frames from the actual
                # encoder output rather than hardcoding, but estimate for construction.
                # mel_len=3000 → conv_out=(3000-1)//2+1=1500 → pool=(1500-2)//2+1=750
                num_frames = 750
                audio_times = torch.stack([
                    torch.arange(num_frames, dtype=torch.float32) * FRAME_DURATION_S + pos
                    for pos in batch_positions
                ]).to(self.device)

                with torch.inference_mode():
                    output = self.model(
                        input_features,
                        input_features_mask=input_features_mask,
                        audio_times=audio_times,
                    )
                    # last_hidden_state: [batch, frames, 1280]
                    hidden = output.last_hidden_state
                    if self._has_projector:
                        hidden = self.projector(hidden)  # [batch, frames, 3584]
                    features = hidden.mean(dim=1)  # Pool time → [batch, dim]
                    all_features.append(features.cpu())

                del input_features, input_features_mask, audio_times, output, hidden, features

            # Average across all chunks
            all_features = torch.cat(all_features, dim=0)
            embedding = torch.mean(all_features, dim=0)
            normalized = F.normalize(embedding, p=2, dim=0)
            return normalized.float().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in inference: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return None

    def generate_from_audio(
        self, waveform: np.ndarray, duration_s: float, filename: str = "<audio>"
    ) -> Optional[list[float]]:
        """
        Generate embedding from a pre-loaded waveform.

        Use this with load_audio() for prefetching (load next file while
        GPU processes current one).

        Returns:
            3584-dim (with projector) or 1280-dim (without) normalized embedding,
            or None on failure.
        """
        self._load_model_if_needed()

        if duration_s < 3.0:
            logger.warning(f"{filename}: too short ({duration_s:.1f}s)")
            return None

        # Select stratified positions
        num_chunks = self._calculate_num_chunks(duration_s)
        positions = self._select_chunk_positions(duration_s, num_chunks)
        chunks = self._extract_chunks(waveform, positions)

        logger.debug(f"{filename}: {len(chunks)} chunks from {duration_s:.1f}s")

        if not chunks:
            return None

        return self._infer(chunks, positions)

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """
        Generate embedding for a single audio file.

        Convenience method that loads audio then generates embedding.
        For better throughput during scanning, use load_audio() +
        generate_from_audio() with prefetching instead.
        """
        result = self.load_audio(filepath)
        if result is None:
            return None
        waveform, duration_s = result
        return self.generate_from_audio(waveform, duration_s, filepath.name)

    def unload_model(self):
        """Free GPU memory by unloading the model, projector, and feature extractor."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.projector is not None:
            del self.projector
            self.projector = None
            self._has_projector = False

        if self.feature_extractor is not None:
            del self.feature_extractor
            self.feature_extractor = None

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

        logger.info("Music Flamingo encoder unloaded")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (3584 with projector, 1280 without)."""
        # Force model load so we know if projector is available
        self._load_model_if_needed()
        return PROJECTED_DIM if self._has_projector else ENCODER_DIM
