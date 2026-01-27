"""MuQ embedding generation for music similarity search."""

import logging
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MuQEmbeddingGenerator:
    """
    Generates audio embeddings using Tencent's MuQ-large-msd-iter model.

    MuQ-large-msd-iter is the SOTA pure music understanding model,
    trained with self-supervised learning on 160K+ hours of music.
    Unlike MuQ-MuLan (designed for music-text retrieval), this model
    is optimized for pure audio similarity tasks.

    Output: 1024-dim embeddings (pooled from temporal features).
    """

    def __init__(
        self,
        model_id: str = "OpenMuQ/MuQ-large-msd-iter",
        target_sr: int = 24000,  # MuQ uses 24kHz
        chunk_duration_s: int = 10,
        base_chunks: int = 3,   # Stratified samples across duration
        contrast_chunks: int = 2,  # High/low energy samples
        max_chunks: int = 20,   # Long tracks (DJ sets): up to 200s coverage
    ):
        self.model_id = model_id
        self.target_sr = target_sr
        self.chunk_duration_s = chunk_duration_s
        self.base_chunks = base_chunks
        self.contrast_chunks = contrast_chunks
        self.max_chunks = max_chunks

        self.device = self._get_best_device()
        logger.info(f"Selected device: {self.device}")

        # Lazy loading
        self.model = None

        # MuQ-large-msd-iter has known NaN issues with FP16, always use FP32
        self.use_half = False
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

    def _scan_energy_profile(self, filepath: Path, duration_s: float) -> list[tuple[float, float]]:
        """
        Quick RMS energy scan. Returns [(position, rms), ...] for each second.

        Uses low sample rate (8kHz) for speed - energy detection doesn't need high fidelity.
        """
        import librosa

        try:
            # Load at low sample rate for speed (~10ms for typical song)
            scan_sr = 8000
            y, _ = librosa.load(str(filepath), sr=scan_sr, mono=True)

            # Calculate RMS energy in 1-second windows
            hop_length = scan_sr  # 1 second hops
            frame_length = scan_sr  # 1 second frames

            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

            # Build profile: (position_in_seconds, rms_value)
            profile = [(i, float(rms[i])) for i in range(len(rms))]
            return profile

        except Exception as e:
            logger.debug(f"Energy scan failed for {filepath.name}: {e}")
            return []

    def _select_contrast_positions(
        self,
        energy_profile: list[tuple[float, float]],
        existing_positions: list[float],
        duration_s: float,
    ) -> list[float]:
        """
        Find highest and lowest energy positions not overlapping existing samples.

        Returns up to contrast_chunks positions (typically 2: one high, one low energy).
        """
        if not energy_profile or self.contrast_chunks == 0:
            return []

        usable = duration_s - self.chunk_duration_s
        if usable <= 0:
            return []

        # Filter to positions that can fit a full chunk
        valid_profile = [(pos, energy) for pos, energy in energy_profile
                         if pos <= usable]

        if not valid_profile:
            return []

        def overlaps_existing(pos: float) -> bool:
            """Check if position overlaps with any existing chunk."""
            for existing in existing_positions:
                # Chunks overlap if they're within chunk_duration of each other
                if abs(pos - existing) < self.chunk_duration_s:
                    return True
            return False

        # Sort by energy to find extremes
        sorted_by_energy = sorted(valid_profile, key=lambda x: x[1])

        contrast_positions = []
        used_positions = list(existing_positions)

        # Try to add lowest energy position
        for pos, _ in sorted_by_energy:
            if not overlaps_existing(pos):
                contrast_positions.append(float(pos))
                used_positions.append(pos)
                break

        # Try to add highest energy position
        for pos, _ in reversed(sorted_by_energy):
            overlap = False
            for used in used_positions:
                if abs(pos - used) < self.chunk_duration_s:
                    overlap = True
                    break
            if not overlap:
                contrast_positions.append(float(pos))
                break

        return contrast_positions[:self.contrast_chunks]

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """Generate embedding for a single audio file."""
        results = self.generate_embedding_batch([filepath])
        return results[0] if results else None

    def generate_embedding_batch(self, filepaths: list[Path]) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple audio files.

        For each file:
        1. Get duration without loading
        2. Scan energy profile (cheap, ~10ms)
        3. Select stratified positions across full duration
        4. Select contrast positions (high/low energy) avoiding overlap
        5. Load each chunk via seeking (memory efficient)
        6. Generate embeddings and average

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

                # Stratified sampling
                num_stratified = self._calculate_num_chunks(duration)
                stratified_positions = self._select_chunk_positions(duration, num_stratified)

                # Contrast sampling (high/low energy)
                energy_profile = self._scan_energy_profile(filepath, duration)
                contrast_positions = self._select_contrast_positions(
                    energy_profile, stratified_positions, duration
                )

                # Combine positions, respecting max_chunks limit
                positions = stratified_positions + contrast_positions
                if len(positions) > self.max_chunks:
                    positions = positions[:self.max_chunks]

                logger.debug(
                    f"{filepath.name}: {len(stratified_positions)} stratified + "
                    f"{len(contrast_positions)} contrast = {len(positions)} chunks"
                )

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

            except Exception as e:
                logger.error(f"Error loading audio file {filepath.name}: {e}")
                file_chunk_counts.append(0)

        if not all_chunks:
            return [None] * len(filepaths)

        try:
            # Process all chunks - MuQ takes raw waveform tensors
            # Stack all chunks into a batch tensor (always FP32 for MuQ stability)
            chunk_tensors = [
                torch.tensor(chunk, dtype=torch.float32) for chunk in all_chunks
            ]
            batch_tensor = torch.stack(chunk_tensors).to(device=self.device)

            with torch.no_grad():
                # MuQ inference: model(tensor) -> output with last_hidden_state
                # Note: MuQ uses positional arg, not wavs= keyword
                # last_hidden_state shape: [batch, time_frames, hidden_dim]
                output = self.model(batch_tensor)

                # Pool temporal dimension to get fixed-size embedding per chunk
                # Mean pooling over time: [batch, time, hidden] -> [batch, hidden]
                audio_features = output.last_hidden_state.mean(dim=1)

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
        """Return the embedding dimension (1024 for MuQ-large-msd-iter)."""
        return 1024
