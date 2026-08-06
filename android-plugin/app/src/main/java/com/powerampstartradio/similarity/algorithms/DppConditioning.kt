package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.EmbeddingIndex
import kotlin.math.sqrt

/** Schur-complement DPP kernel: remove the span of recordings already included in the set. */
internal class DppConditioning private constructor(
    private val basis: List<DoubleArray>,
    private val residualNorms: DoubleArray,
    private val rememberedIds: Set<Long>,
) {
    val remainingDimensions: Int get() = basis.first().size - basis.size
    fun residualNorm(row: Int): Double = residualNorms[row]
    fun remembers(trackId: Long): Boolean = trackId in rememberedIds

    fun project(vector: FloatArray) {
        if (basis.size == vector.size) {
            vector.fill(0f)
            return
        }
        val residual = DoubleArray(vector.size) { vector[it].toDouble() }
        for (direction in basis) {
            var coefficient = 0.0
            for (d in residual.indices) coefficient += residual[d] * direction[d]
            for (d in residual.indices) residual[d] -= coefficient * direction[d]
        }
        for (d in vector.indices) vector[d] = residual[d].toFloat()
    }

    companion object {
        fun create(
            index: EmbeddingIndex,
            trackIds: List<Long>,
            cancellationCheck: (() -> Unit)?,
        ): DppConditioning? {
            if (trackIds.isEmpty()) return null
            val basis = ArrayList<DoubleArray>()
            for (id in trackIds.distinct()) {
                cancellationCheck?.invoke()
                val embedding = index.getEmbeddingByTrackId(id) ?: continue
                val residual = DoubleArray(index.dim) { embedding[it].toDouble() }
                // Reorthogonalization keeps nearly duplicate remembered recordings stable.
                repeat(2) {
                    for (direction in basis) {
                        var coefficient = 0.0
                        for (d in residual.indices) coefficient += residual[d] * direction[d]
                        for (d in residual.indices) residual[d] -= coefficient * direction[d]
                    }
                }
                var squaredNorm = 0.0
                for (v in residual) squaredNorm += v * v
                if (squaredNorm <= 1e-12) continue
                val norm = sqrt(squaredNorm)
                for (d in residual.indices) residual[d] /= norm
                basis += residual
                if (basis.size == index.dim) break
            }
            if (basis.isEmpty()) return null
            if (basis.size == index.dim) {
                return DppConditioning(basis, DoubleArray(index.numTracks), trackIds.toSet())
            }
            val norms = DoubleArray(index.numTracks) { 1.0 }
            val scores = FloatArray(index.numTracks)
            for (direction in basis) {
                index.computeAllSimilaritiesInto(
                    FloatArray(index.dim) { direction[it].toFloat() },
                    scores,
                    cancellationCheck,
                )
                for (row in norms.indices) {
                    val score = scores[row].toDouble()
                    norms[row] -= score * score
                }
            }
            for (row in norms.indices) norms[row] = norms[row].coerceIn(0.0, 1.0)
            return DppConditioning(basis, norms, trackIds.toSet())
        }
    }
}
