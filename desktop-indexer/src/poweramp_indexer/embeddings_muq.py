"""MuQ embedding generation for music similarity search."""

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoadedAudio:
    """Pre-loaded audio data ready for GPU inference."""
    filepath: Path
    waveform: np.ndarray  # Full waveform at target sample rate
    duration_s: float
    energy_profile: list[tuple[float, float]]  # [(position_s, rms), ...]


@dataclass
class PreparedBatch:
    """Batch of audio chunks ready for GPU inference."""
    filepaths: list[Path]
    chunks: list[np.ndarray]  # All chunks from all files
    file_chunk_counts: list[int]  # How many chunks per file


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

    def _load_audio_full(self, filepath: Path) -> Optional[LoadedAudio]:
        """
        Load audio file once and extract all needed information.

        This consolidates what was previously 3+ file reads into 1:
        - Duration (from waveform length)
        - Energy profile (computed from waveform)
        - The waveform itself (for chunk extraction)

        Returns LoadedAudio with all data needed for chunk selection and extraction.
        """
        import librosa

        try:
            # Single load at target sample rate
            waveform, sr = librosa.load(str(filepath), sr=self.target_sr, mono=True)
            duration_s = len(waveform) / sr

            # Compute energy profile from loaded waveform (no second file read)
            energy_profile = self._compute_energy_profile(waveform, sr)

            return LoadedAudio(
                filepath=filepath,
                waveform=waveform,
                duration_s=duration_s,
                energy_profile=energy_profile
            )
        except Exception as e:
            logger.error(f"Error loading {filepath.name}: {e}")
            return None

    def _compute_energy_profile(self, waveform: np.ndarray, sr: int) -> list[tuple[float, float]]:
        """
        Compute RMS energy profile from pre-loaded waveform.

        Returns [(position_in_seconds, rms_value), ...] for each second.
        """
        import librosa

        try:
            # Calculate RMS energy in 1-second windows
            hop_length = sr  # 1 second hops
            frame_length = sr  # 1 second frames

            rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]

            # Build profile: (position_in_seconds, rms_value)
            return [(i, float(rms[i])) for i in range(len(rms))]
        except Exception as e:
            logger.debug(f"Energy profile computation failed: {e}")
            return []

    def _extract_chunks_from_waveform(
        self,
        waveform: np.ndarray,
        positions: list[float]
    ) -> list[np.ndarray]:
        """
        Extract audio chunks from pre-loaded waveform by slicing.

        Much faster than librosa.load() with offset/duration for each chunk.
        """
        import librosa

        chunks = []
        expected_samples = self.chunk_duration_s * self.target_sr

        for pos in positions:
            start_sample = int(pos * self.target_sr)
            end_sample = start_sample + expected_samples

            chunk = waveform[start_sample:end_sample]

            # Ensure exact length for torch.stack compatibility
            if len(chunk) < expected_samples:
                chunk = librosa.util.fix_length(chunk, size=expected_samples)

            chunks.append(chunk)

        return chunks

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

    def _prepare_file(self, filepath: Path) -> tuple[Path, list[np.ndarray], int]:
        """
        Load a single file and prepare its chunks.

        Returns (filepath, chunks, chunk_count) tuple.
        This method is designed to be called from thread pool workers.
        """
        try:
            loaded = self._load_audio_full(filepath)
            if loaded is None:
                return (filepath, [], 0)

            if loaded.duration_s < self.chunk_duration_s:
                logger.warning(
                    f"File {filepath.name} is too short ({loaded.duration_s:.1f}s) "
                    f"for even one chunk of {self.chunk_duration_s}s."
                )
                return (filepath, [], 0)

            # Stratified sampling
            num_stratified = self._calculate_num_chunks(loaded.duration_s)
            stratified_positions = self._select_chunk_positions(loaded.duration_s, num_stratified)

            # Contrast sampling (high/low energy)
            contrast_positions = self._select_contrast_positions(
                loaded.energy_profile, stratified_positions, loaded.duration_s
            )

            # Combine positions, respecting max_chunks limit
            positions = stratified_positions + contrast_positions
            if len(positions) > self.max_chunks:
                positions = positions[:self.max_chunks]

            logger.debug(
                f"{filepath.name}: {len(stratified_positions)} stratified + "
                f"{len(contrast_positions)} contrast = {len(positions)} chunks"
            )

            # Extract chunks from pre-loaded waveform (fast array slicing)
            chunks = self._extract_chunks_from_waveform(loaded.waveform, positions)

            if not chunks:
                logger.warning(f"No valid chunks generated for {filepath.name}.")
                return (filepath, [], 0)

            return (filepath, chunks, len(chunks))

        except Exception as e:
            logger.error(f"Error loading audio file {filepath.name}: {e}")
            return (filepath, [], 0)

    def _prepare_batch(self, filepaths: list[Path], num_workers: int = 8) -> PreparedBatch:
        """
        Prepare a batch of files for GPU inference using parallel loading.

        Uses ThreadPoolExecutor to load multiple files concurrently.
        librosa releases GIL during I/O, so threads work well here.
        """
        all_chunks = []
        file_chunk_counts = []
        results_by_file = {}

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._prepare_file, fp): fp for fp in filepaths}

            for future in as_completed(futures):
                filepath, chunks, chunk_count = future.result()
                results_by_file[filepath] = (chunks, chunk_count)

        # Reassemble in original order
        for filepath in filepaths:
            chunks, chunk_count = results_by_file.get(filepath, ([], 0))
            all_chunks.extend(chunks)
            file_chunk_counts.append(chunk_count)

        return PreparedBatch(
            filepaths=filepaths,
            chunks=all_chunks,
            file_chunk_counts=file_chunk_counts
        )

    def _infer_batch(self, prepared: PreparedBatch) -> list[Optional[list[float]]]:
        """
        Run GPU inference on a prepared batch.

        Takes PreparedBatch with pre-loaded chunks, returns embeddings.
        """
        if not prepared.chunks:
            return [None] * len(prepared.filepaths)

        try:
            # Stack all chunks into a batch tensor (always FP32 for MuQ stability)
            chunk_tensors = [
                torch.tensor(chunk, dtype=torch.float32) for chunk in prepared.chunks
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

            for chunk_count in prepared.file_chunk_counts:
                if chunk_count == 0:
                    results.append(None)
                else:
                    file_features = audio_features[chunk_idx:chunk_idx + chunk_count]
                    mean_embedding = torch.mean(file_features, dim=0)
                    normalized_embedding = F.normalize(mean_embedding, p=2, dim=0)
                    results.append(normalized_embedding.cpu().float().numpy().tolist())
                    chunk_idx += chunk_count

            logger.debug(
                f"Processed batch of {len(prepared.filepaths)} files "
                f"({len(prepared.chunks)} total chunks)"
            )
            return results

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            return [None] * len(prepared.filepaths)

    def generate_embedding(self, filepath: Path) -> Optional[list[float]]:
        """Generate embedding for a single audio file."""
        results = self.generate_embedding_batch([filepath])
        return results[0] if results else None

    def generate_embedding_batch(
        self,
        filepaths: list[Path],
        num_workers: int = 8
    ) -> list[Optional[list[float]]]:
        """
        Generate embeddings for multiple audio files.

        Optimized pipeline:
        1. Load files in parallel using ThreadPoolExecutor
        2. For each file: single load, compute energy, select positions, extract chunks
        3. Run GPU inference on all chunks at once
        4. Average chunk embeddings per file

        Args:
            filepaths: List of audio file paths
            num_workers: Number of parallel file loading threads (default: 4)

        Returns:
            List of embeddings (or None for failed files)
        """
        if not filepaths:
            return []

        self._load_model_if_needed()

        # Prepare batch (parallel file loading)
        prepared = self._prepare_batch(filepaths, num_workers=num_workers)

        # Run GPU inference
        return self._infer_batch(prepared)

    def generate_embeddings_prefetched(
        self,
        all_filepaths: list[Path],
        batch_size: int = 8,
        num_workers: int = 8,
        prefetch_batches: int = 2
    ):
        """
        Generator that yields (filepaths, embeddings) with prefetching.

        Double-buffer pattern: loads batch N+1 while GPU processes batch N.
        This keeps the GPU busy while CPU loads the next batch.

        Args:
            all_filepaths: All files to process
            batch_size: Files per GPU batch
            num_workers: Parallel loading threads
            prefetch_batches: Number of batches to prefetch (default: 2)

        Yields:
            (batch_filepaths, batch_embeddings) tuples
        """
        if not all_filepaths:
            return

        self._load_model_if_needed()

        # Split into batches
        batches = [
            all_filepaths[i:i + batch_size]
            for i in range(0, len(all_filepaths), batch_size)
        ]

        if not batches:
            return

        # Prefetch queue for prepared batches
        prefetch_queue: Queue[Optional[PreparedBatch]] = Queue(maxsize=prefetch_batches)

        def prefetch_worker():
            """Background thread that prepares batches ahead of GPU."""
            for batch_files in batches:
                prepared = self._prepare_batch(batch_files, num_workers=num_workers)
                prefetch_queue.put(prepared)
            # Signal end
            prefetch_queue.put(None)

        # Start prefetch thread
        prefetch_thread = Thread(target=prefetch_worker, daemon=True)
        prefetch_thread.start()

        # Process batches as they become available
        while True:
            prepared = prefetch_queue.get()
            if prepared is None:
                break

            embeddings = self._infer_batch(prepared)
            yield (prepared.filepaths, embeddings)

        prefetch_thread.join()

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
