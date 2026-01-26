"""MuQ-MuLan embedding generation for music similarity search."""

import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MuQEmbeddingGenerator:
    """
    Generates audio embeddings using Tencent's MuQ-MuLan model.

    MuQ-MuLan outperforms CLAP on music understanding benchmarks
    while maintaining the same 512-dim embedding format.
    """

    def __init__(
        self,
        model_id: str = "OpenMuQ/MuQ-MuLan-large",
        target_sr: int = 24000,  # MuQ uses 24kHz (not 48kHz like CLAP)
        chunk_duration_s: int = 10,
        base_chunks: int = 3,
        max_chunks: int = 20,
    ):
        self.model_id = model_id
        self.target_sr = target_sr
        self.chunk_duration_s = chunk_duration_s
        self.base_chunks = base_chunks
        self.max_chunks = max_chunks

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy loading
        self.model = None

        self.use_half = self._can_use_half_precision()
        if self.use_half:
            logger.info("Half precision (FP16) is supported and will be used.")
        else:
            logger.info("Half precision not supported, using full precision (FP32).")

    @staticmethod
    def _get_best_device() -> str:
        """Determine the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _can_use_half_precision(self) -> bool:
        """Check if the device supports half precision."""
        if self.device == "cuda":
            return True
        if self.device == "mps":
            try:
                torch.tensor([1.0], dtype=torch.half).to(self.device)
                return True
            except RuntimeError:
                logger.warning("MPS device does not support half precision")
                return False
        return False

    def _load_model_if_needed(self):
        """Lazy load the MuQ-MuLan model."""
        if self.model is not None:
            return

        from muq import MuQMuLan

        logger.info(f"Loading model '{self.model_id}' to device '{self.device}'...")

        self.model = MuQMuLan.from_pretrained(self.model_id)
        self.model = self.model.to(self.device)

        if self.use_half and self.device == "cuda":
            logger.info("Applying half precision (FP16) to model for CUDA device.")
            self.model.half()

        self.model.eval()
        logger.info("Model loaded successfully.")

    def _get_duration(self, filepath: Path) -> Optional[float]:
        """Get audio duration without loading the file."""
        import librosa
        try:
            return librosa.get_duration(path=str(filepath))
        except Exception as e:
            logger.error(f"Error getting duration for {filepath.name}: {e}")
            return None

    def _calculate_num_chunks(self, duration_s: float) -> int:
        """Scale chunk count based on audio duration."""
        if duration_s < 60:
            return max(1, min(int(duration_s // self.chunk_duration_s), self.base_chunks))
        if duration_s <= 180:
            return self.base_chunks
        if duration_s <= 600:
            return max(self.base_chunks, int(duration_s // 60))
        return min(self.max_chunks, int(duration_s // 120))

    def _select_chunk_positions(self, duration_s: float, num_chunks: int) -> list[float]:
        """Stratified random sampling - one random position per bin across full duration."""
        usable = duration_s - self.chunk_duration_s
        if usable <= 0:
            return [0.0]
        bin_size = usable / num_chunks
        return [
            random.uniform(i * bin_size, min((i + 1) * bin_size, usable))
            for i in range(num_chunks)
        ]

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """Generate embedding for a single audio file."""
        results = self.generate_embedding_batch([filepath])
        return results[0] if results else None

    def generate_embedding_batch(self, filepaths: list[Path]) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple audio files.

        For each file:
        1. Get duration without loading
        2. Calculate number of chunks based on duration
        3. Select stratified positions across full duration
        4. Load each chunk via seeking (memory efficient)
        5. Generate embeddings and average

        Args:
            filepaths: List of audio file paths

        Returns:
            List of embeddings (or None for failed files)
        """
        if not filepaths:
            return []

        import librosa

        self._load_model_if_needed()

        all_chunks = []
        file_chunk_counts = []

        for filepath in filepaths:
            try:
                duration = self._get_duration(filepath)
                if duration is None:
                    file_chunk_counts.append(0)
                    continue

                if duration < self.chunk_duration_s:
                    logger.warning(
                        f"File {filepath.name} is too short ({duration:.1f}s) "
                        f"for even one chunk of {self.chunk_duration_s}s."
                    )
                    file_chunk_counts.append(0)
                    continue

                num_chunks = self._calculate_num_chunks(duration)
                positions = self._select_chunk_positions(duration, num_chunks)

                chunks = []
                expected_samples = self.chunk_duration_s * self.target_sr
                for pos in positions:
                    waveform, _ = librosa.load(
                        str(filepath),
                        sr=self.target_sr,
                        mono=True,
                        offset=pos,
                        duration=self.chunk_duration_s
                    )
                    # Ensure exact length for torch.stack compatibility
                    if len(waveform) < expected_samples:
                        waveform = librosa.util.fix_length(waveform, size=expected_samples)
                    else:
                        waveform = waveform[:expected_samples]
                    chunks.append(waveform)

                if not chunks:
                    logger.warning(f"No valid chunks generated for {filepath.name}.")
                    file_chunk_counts.append(0)
                    continue

                all_chunks.extend(chunks)
                file_chunk_counts.append(len(chunks))
                logger.debug(
                    f"{filepath.name}: {num_chunks} chunks from {duration:.0f}s file"
                )

            except Exception as e:
                logger.error(f"Error loading audio file {filepath.name}: {e}")
                file_chunk_counts.append(0)

        if not all_chunks:
            return [None] * len(filepaths)

        try:
            # Process all chunks - MuQ-MuLan takes raw tensors
            model_dtype = next(self.model.parameters()).dtype

            # Stack all chunks into a batch tensor
            chunk_tensors = [
                torch.tensor(chunk, dtype=torch.float32) for chunk in all_chunks
            ]
            batch_tensor = torch.stack(chunk_tensors).to(device=self.device, dtype=model_dtype)

            with torch.no_grad():
                # MuQ-MuLan inference: model(wavs=tensor) -> [batch, 512]
                audio_features = self.model(wavs=batch_tensor)

            # Split results back to individual files and compute mean embeddings
            results = []
            chunk_idx = 0

            for chunk_count in file_chunk_counts:
                if chunk_count == 0:
                    results.append(None)
                else:
                    file_features = audio_features[chunk_idx:chunk_idx + chunk_count]
                    mean_embedding = torch.mean(file_features, dim=0)
                    normalized_embedding = F.normalize(mean_embedding, p=2, dim=0)
                    results.append(normalized_embedding.cpu().float().numpy().tolist())
                    chunk_idx += chunk_count

            logger.debug(
                f"Processed batch of {len(filepaths)} files ({len(all_chunks)} total chunks)"
            )
            return results

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return [None] * len(filepaths)

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
        """Return the embedding dimension (512 for MuQ-MuLan)."""
        return 512
