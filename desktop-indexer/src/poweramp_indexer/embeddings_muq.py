"""MuQ embedding generation for music similarity search."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MuQEmbeddingGenerator:
    """
    Generates audio embeddings using Tencent's MuQ-large-msd-iter model.

    MuQ-large-msd-iter is a music understanding model trained with
    self-supervised learning on the Million Song Dataset. Unlike
    MuQ-MuLan (designed for music-text retrieval), this model is
    optimized for pure audio similarity tasks.

    Output: 1024-dim embeddings (pooled from temporal features).
    """

    def __init__(
        self,
        model_id: str = "OpenMuQ/MuQ-large-msd-iter",
        target_sr: int = 24000,  # MuQ uses 24kHz
        chunk_duration_s: int = 30,  # MuQ optimal input length
        max_chunks: int = 30,   # 30 min coverage (1 chunk per minute)
    ):
        self.model_id = model_id
        self.target_sr = target_sr
        self.chunk_duration_s = chunk_duration_s
        self.max_chunks = max_chunks

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy loading
        self.model = None

        # MuQ-large-msd-iter has known NaN issues with FP16, always use FP32
        logger.info("Using full precision (FP32) for MuQ model stability.")

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model_if_needed(self):
        """Lazy load the MuQ model."""
        if self.model is not None:
            return

        from muq import MuQ

        logger.info(f"Loading model '{self.model_id}' to device '{self.device}'...")

        self.model = MuQ.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _load_audio(self, filepath: Path) -> Optional[tuple[np.ndarray, float]]:
        """
        Load audio file.

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
        """1 chunk per minute, max 30 (covers 30 min)."""
        minutes = duration_s / 60
        return max(1, min(int(minutes), self.max_chunks))

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

    def _infer(self, chunks: list[np.ndarray], max_batch: int = 5) -> Optional[list[float]]:
        """
        Run GPU inference on chunks and return averaged, normalized embedding.

        Processes in sub-batches of max_batch to avoid OOM on long tracks.
        """
        try:
            all_features = []

            # Process in sub-batches to avoid OOM
            for i in range(0, len(chunks), max_batch):
                batch_chunks = chunks[i:i + max_batch]
                batch = torch.stack([
                    torch.tensor(c, dtype=torch.float32) for c in batch_chunks
                ]).to(self.device)

                with torch.no_grad():
                    output = self.model(batch)
                    features = output.last_hidden_state.mean(dim=1)  # Pool time dimension
                    all_features.append(features.cpu())

                # Cleanup between sub-batches
                del batch, output, features
                if self.device == "cuda":
                    torch.cuda.empty_cache()

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

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """
        Generate embedding for a single audio file.

        Pipeline: load audio -> select chunk positions -> extract chunks -> GPU inference
        """
        self._load_model_if_needed()

        # Load audio
        result = self._load_audio(filepath)
        if result is None:
            return None
        waveform, duration_s = result

        if duration_s < self.chunk_duration_s:
            logger.warning(f"{filepath.name}: too short ({duration_s:.1f}s)")
            return None

        # Select stratified positions
        num_chunks = self._calculate_num_chunks(duration_s)
        positions = self._select_chunk_positions(duration_s, num_chunks)

        logger.debug(f"{filepath.name}: {len(positions)} chunks from {duration_s:.1f}s")

        # Extract chunks then free waveform
        chunks = self._extract_chunks(waveform, positions)
        del waveform  # Free memory before inference

        if not chunks:
            return None

        return self._infer(chunks)

    def unload_model(self):
        """Free GPU memory by unloading the model."""
        if self.model is not None:
            del self.model
            self.model = None

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

            logger.info("Model unloaded")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (1024 for MuQ-large-msd-iter)."""
        return 1024
