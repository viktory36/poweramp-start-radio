package com.powerampstartradio.indexing

import android.util.Log
import java.io.File

/**
 * Orchestrates CLaMP3 audio embedding generation for on-device indexing.
 *
 * The CLaMP3 pipeline has two GPU phases (Adreno can't run two OpenCL contexts):
 * 1. MERT: raw waveform → 768d feature vectors per 5-second window
 * 2. CLaMP3 audio encoder: aggregates MERT features → single 768d embedding
 *
 * IndexingService handles the two-phase GPU lifecycle (load MERT → process all
 * tracks → close → load CLaMP3 → encode all). This class provides the per-track
 * processing logic for each phase.
 */
class EmbeddingProcessor(
    private val audioDecoder: AudioDecoder = AudioDecoder(),
) {

    companion object {
        private const val TAG = "EmbeddingProcessor"
        const val EMBEDDING_DIM = 768
    }

    /**
     * Decode audio for MERT processing (24kHz mono).
     *
     * @param audioFile Path to the audio file
     * @param maxDurationS Maximum duration to decode (0 = unlimited)
     * @return Decoded audio at 24kHz, or null on failure
     */
    fun decodeForMert(
        audioFile: File,
        maxDurationS: Int = MertInference.CHUNK_DURATION_S,
    ): AudioDecoder.DecodedAudio? {
        return audioDecoder.decode(audioFile, MertInference.SAMPLE_RATE, maxDurationS = maxDurationS)
    }
}
