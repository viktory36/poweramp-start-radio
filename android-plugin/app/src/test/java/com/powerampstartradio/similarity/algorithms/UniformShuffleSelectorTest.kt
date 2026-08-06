package com.powerampstartradio.similarity.algorithms

import com.powerampstartradio.data.UniformShuffleIdentityKey
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class UniformShuffleSelectorTest {
    @Test
    fun sameSeedAndLibraryProduceExactKnownPermutation() {
        val selected = UniformShuffleSelector.select(
            trackIds = longArrayOf(10, 20, 30, 40, 50, 60),
            numSelect = 6,
            seed = 0x1234,
            identityKeyForTrack = IDENTITY,
        )

        assertEquals(listOf(60L, 50L, 30L, 40L, 20L, 10L), selected.map { it.trackId })
        assertEquals(listOf(1, 2, 3, 4, 5, 6), selected.map { it.shuffleRank })
    }

    @Test
    fun permutationDoesNotDependOnDatabaseIterationOrder() {
        val forward = UniformShuffleSelector.select(
            longArrayOf(1, 2, 3, 4, 5, 6, 7),
            numSelect = 7,
            seed = 99,
            identityKeyForTrack = IDENTITY,
        )
        val reverse = UniformShuffleSelector.select(
            longArrayOf(7, 6, 5, 4, 3, 2, 1),
            numSelect = 7,
            seed = 99,
            identityKeyForTrack = IDENTITY,
        )

        assertEquals(forward, reverse)
    }

    @Test
    fun changingTheRecordedSeedChangesThePermutation() {
        val ids = LongArray(40) { it + 1L }
        val first = UniformShuffleSelector.select(
            ids, numSelect = 40, seed = 1, identityKeyForTrack = IDENTITY,
        )
        val second = UniformShuffleSelector.select(
            ids, numSelect = 40, seed = 2, identityKeyForTrack = IDENTITY,
        )

        assertNotEquals(first.map { it.trackId }, second.map { it.trackId })
    }

    @Test
    fun nextSeedDoesNotTranslateTheConsecutiveIdPermutation() {
        val ids = LongArray(1_000) { it + 1L }
        val seed = 0x1234L
        val first = UniformShuffleSelector.select(
            ids, numSelect = 100, seed = seed, identityKeyForTrack = IDENTITY,
        )
            .map { it.trackId }
        val second = UniformShuffleSelector.select(
            ids,
            numSelect = 100,
            seed = UniformShuffleSelector.nextSeed(seed),
            identityKeyForTrack = IDENTITY,
        ).map { it.trackId }

        assertNotEquals(first, second)
        assertNotEquals(first.drop(1), second.dropLast(1))
    }

    @Test
    fun exclusionsAndEligibilityAreAppliedDuringSelectionAndRefill() {
        val unconstrained = UniformShuffleSelector.select(
            LongArray(20) { it + 1L },
            numSelect = 20,
            seed = 73,
            identityKeyForTrack = IDENTITY,
            excludeIds = setOf(5L),
        )
        val rejected = unconstrained[1].trackId

        val constrained = UniformShuffleSelector.select(
            LongArray(20) { it + 1L },
            numSelect = 5,
            seed = 73,
            identityKeyForTrack = IDENTITY,
            excludeIds = setOf(5L),
            isEligible = { trackId, _ -> trackId != rejected },
        )

        assertEquals(5, constrained.size)
        assertEquals(unconstrained.map { it.trackId }.filterNot { it == rejected }.take(5),
            constrained.map { it.trackId })
        assertEquals(unconstrained[2].shuffleRank, constrained[1].shuffleRank)
    }

    @Test
    fun equivalentRowsAppearOnceAndExcludingOneExcludesTheAcousticIdentity() {
        val identities: (Long) -> UniformShuffleIdentityKey = { trackId ->
            val acousticId = if (trackId == 2L) 1L else trackId
            UniformShuffleIdentityKey(
                high = acousticId,
                low = acousticId.inv(),
                identityToken = "recording:$acousticId",
                isStableAcrossGenerations = true,
            )
        }

        val selected = UniformShuffleSelector.select(
            trackIds = longArrayOf(1, 2, 3),
            numSelect = 3,
            seed = 9,
            identityKeyForTrack = identities,
        )
        val excluded = UniformShuffleSelector.select(
            trackIds = longArrayOf(1, 2, 3),
            numSelect = 3,
            seed = 9,
            identityKeyForTrack = identities,
            excludeIds = setOf(2L),
        )

        assertEquals(setOf(1L, 3L), selected.map { it.trackId }.toSet())
        assertEquals(listOf(3L), excluded.map { it.trackId })
    }

    companion object {
        private val IDENTITY: (Long) -> UniformShuffleIdentityKey = { trackId ->
            UniformShuffleIdentityKey(
                high = trackId,
                low = trackId.inv(),
                identityToken = "track:$trackId",
                isStableAcrossGenerations = true,
            )
        }
    }
}
