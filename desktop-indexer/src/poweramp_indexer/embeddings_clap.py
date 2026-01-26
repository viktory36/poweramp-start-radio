"""CLAP embedding generation - adapted from Mycelium's CLAP adapter."""

import logging
import random
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class CLAPEmbeddingGenerator:
    """
    Generates audio embeddings using LAION's CLAP model.

    Adapted from Mycelium's infrastructure/clap_adapter.py
    """

    def __init__(
        self,
        model_id: str = "laion/larger_clap_music_and_speech",
        target_sr: int = 48000,
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
        self.processor = None

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
        """Lazy load the CLAP model."""
        if self.model is not None:
            return

        from transformers import ClapModel, ClapProcessor

        logger.info(f"Loading model '{self.model_id}' to device '{self.device}'...")

        self.model = ClapModel.from_pretrained(self.model_id, use_safetensors=True).to(self.device)
        self.processor = ClapProcessor.from_pretrained(self.model_id)

        if self.use_half and self.device == "cuda":
            logger.info("Applying half precision (FP16) to model for CUDA device.")
            self.model.half()

        self.model.eval()
        logger.info("Model loaded successfully.")

    def _prepare_inputs(self, inputs: dict) -> dict:
        """Move inputs to the correct device and dtype."""
        model_dtype = next(self.model.parameters()).dtype
        prepared = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    prepared[k] = v.to(device=self.device, dtype=model_dtype)
                else:
                    prepared[k] = v.to(device=self.device)
            else:
                prepared[k] = v
        return prepared

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
                for pos in positions:
                    waveform, _ = librosa.load(
                        str(filepath),
                        sr=self.target_sr,
                        mono=True,
                        offset=pos,
                        duration=self.chunk_duration_s
                    )
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
            # Process all chunks in a single batch
            inputs = self.processor(
                audios=all_chunks,
                sampling_rate=self.target_sr,
                return_tensors="pt",
                padding=True
            )

            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)

            # Split results back to individual files and compute mean embeddings
            results = []
            chunk_idx = 0

            for chunk_count in file_chunk_counts:
                if chunk_count == 0:
                    results.append(None)
                else:
                    file_features = audio_features[chunk_idx:chunk_idx + chunk_count]
                    mean_embedding = torch.mean(file_features, dim=0)
                    normalized_embedding = torch.nn.functional.normalize(mean_embedding, p=2, dim=0)
                    results.append(normalized_embedding.cpu().numpy().tolist())
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
            del self.processor
            self.model = None
            self.processor = None

            if self.device == "cuda":
                torch.cuda.empty_cache()
            elif self.device == "mps":
                torch.mps.empty_cache()

            logger.info("Model unloaded")

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (512 for CLAP)."""
        return 512
