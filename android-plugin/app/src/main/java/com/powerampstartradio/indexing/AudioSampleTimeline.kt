package com.powerampstartradio.indexing

import kotlin.math.round

/** Exact conversion between media microsecond timestamps and PCM sample coordinates. */
internal object AudioSampleTimeline {
    fun sampleAtOrAfter(timeUs: Long, sampleRate: Int): Long {
        require(timeUs >= 0L) { "timeUs must be non-negative" }
        require(sampleRate > 0) { "sampleRate must be positive" }
        val whole = Math.multiplyExact(timeUs / MICROS_PER_SECOND, sampleRate.toLong())
        val remainderProduct = (timeUs % MICROS_PER_SECOND) * sampleRate
        val partial = ceilDiv(remainderProduct, MICROS_PER_SECOND)
        return Math.addExact(whole, partial)
    }

    /** MediaCodec timestamps are rounded representations of an existing sample index. */
    fun nearestSampleForTimestamp(timeUs: Long, sampleRate: Int): Long {
        require(sampleRate > 0) { "sampleRate must be positive" }
        return round(timeUs.toDouble() * sampleRate / MICROS_PER_SECOND).toLong()
    }

    fun spanSampleCount(startUs: Long, durationUs: Long, sampleRate: Int): Long {
        require(durationUs > 0L) { "durationUs must be positive" }
        val endUs = Math.addExact(startUs, durationUs)
        return sampleAtOrAfter(endUs, sampleRate) - sampleAtOrAfter(startUs, sampleRate)
    }

    fun resampledSampleCount(totalInputSamples: Long, fromRate: Int, toRate: Int): Long {
        require(totalInputSamples >= 0L) { "totalInputSamples must be non-negative" }
        require(fromRate > 0 && toRate > 0) { "sample rates must be positive" }
        val divisor = gcd(fromRate, toRate)
        val up = (toRate / divisor).toLong()
        val down = (fromRate / divisor).toLong()
        val whole = Math.multiplyExact(totalInputSamples / down, up)
        val partial = ceilDiv((totalInputSamples % down) * up, down)
        return Math.addExact(whole, partial)
    }

    private fun gcd(first: Int, second: Int): Int {
        var a = first
        var b = second
        while (b != 0) {
            val remainder = a % b
            a = b
            b = remainder
        }
        return a
    }

    private fun ceilDiv(value: Long, divisor: Long): Long =
        if (value == 0L) 0L else 1L + (value - 1L) / divisor

    private const val MICROS_PER_SECOND = 1_000_000L
}
