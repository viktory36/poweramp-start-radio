package com.powerampstartradio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlin.math.roundToInt

/**
 * A horizontal segmented bar where each segment represents a seed's weight.
 * Drag the dividers between segments to redistribute weights.
 * All weights are positive and always sum to 1.0.
 *
 * @param segments List of (label, weight, color) for each seed
 * @param onWeightsChanged Called with new weight list when a divider is dragged
 * @param minWeight Minimum weight per segment (prevents zeroing out)
 */
@Composable
fun ProportionBar(
    segments: List<ProportionSegment>,
    onWeightsChanged: (List<Float>) -> Unit,
    modifier: Modifier = Modifier,
    minWeight: Float = 0.05f,
) {
    if (segments.size < 2) return

    val weights = segments.map { it.weight }
    var barWidthPx by remember { mutableIntStateOf(0) }
    val density = LocalDensity.current

    Box(
        modifier = modifier
            .fillMaxWidth()
            .height(36.dp)
            .clip(RoundedCornerShape(8.dp))
            .onSizeChanged { barWidthPx = it.width }
    ) {
        // Draw segments
        Row(modifier = Modifier.fillMaxSize()) {
            for (i in segments.indices) {
                val seg = segments[i]
                Box(
                    modifier = Modifier
                        .weight(seg.weight.coerceAtLeast(0.001f))
                        .fillMaxHeight()
                        .background(seg.color),
                    contentAlignment = Alignment.Center,
                ) {
                    val pct = (seg.weight * 100).roundToInt()
                    // Only show label if segment is wide enough
                    if (seg.weight > 0.12f) {
                        Text(
                            text = "${seg.label} ${pct}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.White,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.padding(horizontal = 4.dp),
                        )
                    } else if (seg.weight > 0.06f) {
                        Text(
                            text = "${pct}%",
                            style = MaterialTheme.typography.labelSmall,
                            color = Color.White,
                            maxLines = 1,
                        )
                    }
                }
            }
        }

        // Invisible drag handles at each divider boundary
        var cumulativeWeight = 0f
        for (i in 0 until segments.size - 1) {
            cumulativeWeight += weights[i]
            val dividerFraction = cumulativeWeight
            val currentWeights = rememberUpdatedState(weights)

            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .width(24.dp) // touch target
                    .offset(
                        x = with(density) {
                            (barWidthPx * dividerFraction).toDp() - 12.dp
                        }
                    )
                    .pointerInput(i) {
                        awaitEachGesture {
                            val down = awaitFirstDown(requireUnconsumed = false)
                            down.consume()
                            while (true) {
                                val event = awaitPointerEvent()
                                val change = event.changes.firstOrNull() ?: break
                                if (!change.pressed) break
                                val dx = change.positionChange().x
                                if (dx != 0f && barWidthPx > 0) {
                                    change.consume()
                                    val delta = dx / barWidthPx
                                    val ws = currentWeights.value.toMutableList()

                                    // Move weight between segment i and i+1
                                    val leftNew = (ws[i] + delta).coerceIn(minWeight, 1f - minWeight * (ws.size - 1))
                                    val rightNew = (ws[i + 1] - delta).coerceIn(minWeight, 1f - minWeight * (ws.size - 1))

                                    // Only apply if both sides stay valid
                                    if (leftNew >= minWeight && rightNew >= minWeight) {
                                        ws[i] = leftNew
                                        ws[i + 1] = rightNew
                                        onWeightsChanged(ws)
                                    }
                                }
                            }
                        }
                    }
            )
        }
    }
}

data class ProportionSegment(
    val label: String,
    val weight: Float,
    val color: Color,
)
