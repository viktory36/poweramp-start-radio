"""Dual-model embedding generation using MuQ and MuLan from identical audio chunks."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DualEmbeddingGenerator:
    """
    Generates audio embeddings using both MuQ and MuLan models from the same audio.

    Sampling strategy:
    - 1 x 30s chunk per minute of audio (stratified across duration)
    - Cap at 30 chunks (covers 30 min of content)
    - MuQ processes full 30s chunks -> 1024-dim embedding
    - MuLan receives 30s chunks, internally splits to 10s clips -> 512-dim embedding

    This enables A/B comparison between models using identical audio content.
    """

    def __init__(
        self,
        muq_model_id: str = "OpenMuQ/MuQ-large-msd-iter",
        mulan_model_id: str = "OpenMuQ/MuQ-MuLan-large",
        target_sr: int = 24000,  # Both models use 24kHz
        chunk_duration_s: int = 30,  # MuQ optimal; MuLan internally splits to 10s
        max_chunks: int = 30,  # 30 min coverage
    ):
        self.muq_model_id = muq_model_id
        self.mulan_model_id = mulan_model_id
        self.target_sr = target_sr
        self.chunk_duration_s = chunk_duration_s
        self.max_chunks = max_chunks

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy loading
        self.muq_model = None
        self.mulan_model = None

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_muq_if_needed(self):
        """Lazy load the MuQ model."""
        if self.muq_model is not None:
            return

        from muq import MuQ

        logger.info(f"Loading MuQ model '{self.muq_model_id}'...")
        self.muq_model = MuQ.from_pretrained(self.muq_model_id)
        self.muq_model = self.muq_model.to(self.device)
        self.muq_model.eval()
        logger.info("MuQ model loaded successfully.")

    def _load_mulan_if_needed(self):
        """Lazy load the MuLan model."""
        if self.mulan_model is not None:
            return

        from muq import MuQMuLan

        logger.info(f"Loading MuLan model '{self.mulan_model_id}'...")
        self.mulan_model = MuQMuLan.from_pretrained(self.mulan_model_id)
        self.mulan_model = self.mulan_model.to(self.device)
        self.mulan_model.eval()
        logger.info("MuLan model loaded successfully.")

    def load_audio(self, filepath: Path) -> Optional[tuple[np.ndarray, float]]:
        """
        Load and resample audio file to 24kHz mono.

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

    def _extract_chunks(
        self, waveform: np.ndarray, positions: list[float]
    ) -> list[np.ndarray]:
        """Extract 30s audio chunks from waveform."""
        import librosa

        chunks = []
        expected_samples = self.chunk_duration_s * self.target_sr

        for pos in positions:
            start = int(pos * self.target_sr)
            chunk = waveform[start : start + expected_samples]
            if len(chunk) < expected_samples:
                chunk = librosa.util.fix_length(chunk, size=expected_samples)
            chunks.append(chunk)

        return chunks

    def _infer_muq(self, chunks: list[np.ndarray], max_batch: int = 5) -> Optional[list[float]]:
        """
        Run MuQ inference on 30s chunks, return averaged 1024-dim embedding.

        MuQ API: output = muq(wavs) where wavs is [batch, samples]
        Returns output.last_hidden_state of shape [batch, time, 1024]
        """
        if not chunks:
            logger.warning("No chunks to process for MuQ")
            return None

        self._load_muq_if_needed()

        try:
            all_features = []

            for i in range(0, len(chunks), max_batch):
                batch_chunks = chunks[i : i + max_batch]
                # MuQ expects [batch, samples] tensor
                batch = torch.stack(
                    [torch.tensor(c, dtype=torch.float32) for c in batch_chunks]
                ).to(self.device)

                with torch.no_grad():
                    output = self.muq_model(batch)
                    # Pool over time dimension: [batch, time, 1024] -> [batch, 1024]
                    features = output.last_hidden_state.mean(dim=1)
                    all_features.append(features.cpu())

                del batch, output, features

            # Average across all chunks: [total_chunks, 1024] -> [1024]
            all_features = torch.cat(all_features, dim=0)
            embedding = torch.mean(all_features, dim=0)
            normalized = F.normalize(embedding, p=2, dim=0)
            return normalized.float().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in MuQ inference: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return None

    def _infer_mulan(
        self, chunks: list[np.ndarray], max_batch: int = 5
    ) -> Optional[list[float]]:
        """
        Run MuLan inference on 30s chunks, return averaged 512-dim embedding.

        MuLan internally splits each chunk into 10s clips via _get_all_clips().
        API: audio_embeds = mulan(wavs=wavs) where wavs is [batch, samples]
        Returns embedding tensor of shape [batch, 512]
        """
        if not chunks:
            logger.warning("No chunks to process for MuLan")
            return None

        self._load_mulan_if_needed()

        try:
            all_features = []

            for i in range(0, len(chunks), max_batch):
                batch_chunks = chunks[i : i + max_batch]
                # MuLan expects wavs=[batch, samples] tensor
                batch = torch.stack(
                    [torch.tensor(c, dtype=torch.float32) for c in batch_chunks]
                ).to(self.device)

                with torch.no_grad():
                    # MuLan API: mulan(wavs=tensor) returns audio embeddings
                    audio_embeds = self.mulan_model(wavs=batch)
                    all_features.append(audio_embeds.cpu())

                del batch, audio_embeds

            # Average across all chunks: [total_chunks, 512] -> [512]
            all_features = torch.cat(all_features, dim=0)
            embedding = torch.mean(all_features, dim=0)
            normalized = F.normalize(embedding, p=2, dim=0)
            return normalized.float().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in MuLan inference: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return None

    def _prepare_chunks(
        self, waveform: np.ndarray, duration_s: float, filename: str
    ) -> Optional[list[np.ndarray]]:
        """Extract chunks from waveform for inference. Returns None if audio is unsuitable."""
        if duration_s < self.chunk_duration_s:
            logger.warning(f"{filename}: too short ({duration_s:.1f}s < 30s)")
            return None

        num_chunks = self._calculate_num_chunks(duration_s)
        positions = self._select_chunk_positions(duration_s, num_chunks)
        chunks = self._extract_chunks(waveform, positions)

        logger.debug(f"{filename}: {len(chunks)} x 30s chunks from {duration_s:.1f}s")

        if not chunks:
            logger.warning(f"{filename}: no chunks extracted")
            return None
        return chunks

    def generate_muq_from_audio(
        self, waveform: np.ndarray, duration_s: float, filename: str = "<audio>"
    ) -> Optional[list[float]]:
        """Generate MuQ embedding only from pre-loaded waveform."""
        chunks = self._prepare_chunks(waveform, duration_s, filename)
        if chunks is None:
            return None
        return self._infer_muq(chunks)

    def generate_mulan_from_audio(
        self, waveform: np.ndarray, duration_s: float, filename: str = "<audio>"
    ) -> Optional[list[float]]:
        """Generate MuLan embedding only from pre-loaded waveform."""
        chunks = self._prepare_chunks(waveform, duration_s, filename)
        if chunks is None:
            return None
        return self._infer_mulan(chunks)

    def generate_from_audio(
        self, waveform: np.ndarray, duration_s: float, filename: str = "<audio>"
    ) -> tuple[Optional[list[float]], Optional[list[float]]]:
        """
        Generate embeddings from a pre-loaded waveform using both models.

        Returns:
            Tuple of (muq_embedding, mulan_embedding).
            Either may be None if that model's inference fails.
        """
        chunks = self._prepare_chunks(waveform, duration_s, filename)
        if chunks is None:
            return None, None

        muq_embedding = self._infer_muq(chunks)
        mulan_embedding = self._infer_mulan(chunks)
        del chunks

        return muq_embedding, mulan_embedding

    def generate_embeddings(
        self, filepath: Path
    ) -> tuple[Optional[list[float]], Optional[list[float]]]:
        """
        Generate embeddings for a single audio file using both models.

        Convenience method that loads audio then generates embeddings.
        For better throughput during scanning, use load_audio() +
        generate_from_audio() with prefetching instead.
        """
        result = self.load_audio(filepath)
        if result is None:
            return None, None
        waveform, duration_s = result
        return self.generate_from_audio(waveform, duration_s, filepath.name)

    def embed_text(self, text: str) -> Optional[list[float]]:
        """
        Generate text embedding using MuLan for text-to-music search.

        MuLan API: text_embeds = mulan(texts=["query"]) returns [1, 512]

        Args:
            text: Query text (e.g., "sufi music", "upbeat electronic")

        Returns:
            512-dim normalized embedding or None on failure.
        """
        self._load_mulan_if_needed()

        try:
            with torch.no_grad():
                # MuLan API: mulan(texts=list) returns text embeddings
                text_embeds = self.mulan_model(texts=[text])
                # text_embeds shape: [1, 512] -> [512]
                embedding = text_embeds[0]
                normalized = F.normalize(embedding, p=2, dim=0)
                return normalized.cpu().float().numpy().tolist()

        except Exception as e:
            logger.error(f"Error in text embedding: {e}")
            return None

    def unload_models(self):
        """Free GPU memory by unloading both models."""
        if self.muq_model is not None:
            del self.muq_model
            self.muq_model = None
            logger.info("MuQ model unloaded")

        if self.mulan_model is not None:
            del self.mulan_model
            self.mulan_model = None
            logger.info("MuLan model unloaded")

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()

    def unload_muq(self):
        """Free GPU memory by unloading MuQ model only."""
        if self.muq_model is not None:
            del self.muq_model
            self.muq_model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("MuQ model unloaded")

    def unload_mulan(self):
        """Free GPU memory by unloading MuLan model only."""
        if self.mulan_model is not None:
            del self.mulan_model
            self.mulan_model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("MuLan model unloaded")

    @property
    def muq_embedding_dim(self) -> int:
        """Return MuQ embedding dimension (1024)."""
        return 1024

    @property
    def mulan_embedding_dim(self) -> int:
        """Return MuLan embedding dimension (512)."""
        return 512
