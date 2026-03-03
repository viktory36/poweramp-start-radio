package com.powerampstartradio.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Circular knob for 0.0 to 1.0 with sign and lock.
 *
 * 270° arc: 7 o'clock (0) → 5 o'clock (1.0). Fills the textbox height.
 * Drag horizontally to adjust. Tap to toggle sign (+/−).
 * Long press to toggle lock. When locked, drags are ignored and knob is dimmed.
 *
 * Percentage label is drawn inside the arc gap (between 5 and 7 o'clock).
 */
@Composable
fun RotaryKnob(
    value: Float,
    onValueChange: (Float) -> Unit,
    onDragEnd: () -> Unit,
    negative: Boolean,
    onToggleSign: () -> Unit,
    locked: Boolean,
    onToggleLock: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val trackColor = MaterialTheme.colorScheme.surfaceVariant
    val positiveColor = MaterialTheme.colorScheme.primary
    val negativeColor = MaterialTheme.colorScheme.error
    val activeColor = when {
        locked -> MaterialTheme.colorScheme.outline.copy(alpha = 0.5f)
        negative -> negativeColor
        else -> positiveColor
    }
    val lockColor = if (locked)
        MaterialTheme.colorScheme.outline
    else
        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.3f)

    val pct = (value * 100).roundToInt()
    val sign = if (negative) "\u2212" else ""
    val displayText = "$sign${pct}%"

    val labelColor = when {
        locked -> MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
        negative -> negativeColor
        else -> MaterialTheme.colorScheme.onSurfaceVariant
    }

    val currentValue = rememberUpdatedState(value)
    val currentLocked = rememberUpdatedState(locked)
    val textMeasurer = rememberTextMeasurer()
    val labelStyle = TextStyle(
        fontSize = 9.sp,
        color = labelColor,
    )

    Canvas(
        modifier = modifier
            .size(56.dp)
            .pointerInput(Unit) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    down.consume()
                    var totalDx = 0f
                    var totalDy = 0f
                    var isTap = true
                    val tapThreshold = 12.dp.toPx()
                    val longPressMs = 400L
                    var longPressFired = false
                    // Launch timer on Main dispatcher for long-press detection
                    val timerJob = CoroutineScope(Dispatchers.Main).launch {
                        delay(longPressMs)
                        if (isTap) {
                            longPressFired = true
                            onToggleLock()
                        }
                    }
                    while (true) {
                        val event = awaitPointerEvent()
                        val change = event.changes.firstOrNull() ?: break
                        if (!change.pressed) break
                        val dx = change.positionChange().x
                        val dy = change.positionChange().y
                        totalDx += dx
                        totalDy += dy
                        if (sqrt(totalDx * totalDx + totalDy * totalDy) > tapThreshold) {
                            isTap = false
                            timerJob?.cancel()
                        }
                        if (!isTap && !currentLocked.value && dx != 0f) {
                            change.consume()
                            val delta = dx / 150f
                            val newVal = (currentValue.value + delta).coerceIn(0f, 1f)
                            onValueChange(newVal)
                        }
                    }
                    timerJob?.cancel()
                    if (isTap && !longPressFired) {
                        onToggleSign()
                    } else if (!isTap) {
                        onDragEnd()
                    }
                }
            },
    ) {
        val cx = size.width / 2
        val cy = size.height / 2
        // Radius fills the square tightly; indicator dot may slightly clip at top
        val r = size.height / 2 - 2.dp.toPx()
        val stroke = 2.5.dp.toPx()

        // Background track
        drawArc(
            color = trackColor,
            startAngle = 135f,
            sweepAngle = 270f,
            useCenter = false,
            topLeft = Offset(cx - r, cy - r),
            size = androidx.compose.ui.geometry.Size(r * 2, r * 2),
            style = Stroke(width = stroke, cap = StrokeCap.Round),
        )

        // Active arc
        val v = currentValue.value
        if (v > 0.01f) {
            drawArc(
                color = activeColor,
                startAngle = 135f,
                sweepAngle = v * 270f,
                useCenter = false,
                topLeft = Offset(cx - r, cy - r),
                size = androidx.compose.ui.geometry.Size(r * 2, r * 2),
                style = Stroke(width = stroke, cap = StrokeCap.Round),
            )
        }

        // Indicator dot
        val angleDeg = 135f + v * 270f
        val angleRad = angleDeg * PI.toFloat() / 180f
        drawCircle(
            color = activeColor,
            radius = 3.dp.toPx(),
            center = Offset(cx + r * cos(angleRad), cy + r * sin(angleRad)),
        )

        // Center icon: lock when locked, +/− sign when unlocked
        if (locked) {
            val lockR = 4.dp.toPx()
            drawCircle(
                color = lockColor,
                radius = lockR,
                center = Offset(cx, cy + 1.dp.toPx()),
            )
            drawArc(
                color = lockColor,
                startAngle = 180f,
                sweepAngle = 180f,
                useCenter = false,
                topLeft = Offset(cx - lockR * 0.6f, cy - lockR * 1.3f),
                size = androidx.compose.ui.geometry.Size(lockR * 1.2f, lockR * 1.2f),
                style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round),
            )
        } else {
            val signLen = 5.dp.toPx()
            val signColor = activeColor
            // Horizontal bar (both + and −)
            drawLine(signColor, Offset(cx - signLen, cy), Offset(cx + signLen, cy),
                strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
            if (!negative) {
                // Vertical bar (+ only)
                drawLine(signColor, Offset(cx, cy - signLen), Offset(cx, cy + signLen),
                    strokeWidth = 2.dp.toPx(), cap = StrokeCap.Round)
            }
        }

        // Percentage label snug in the arc gap (between 5 and 7 o'clock)
        val textResult = textMeasurer.measure(displayText, labelStyle)
        drawText(
            textResult,
            topLeft = Offset(
                (size.width - textResult.size.width) / 2f,
                cy + r * 0.7f,
            ),
        )
    }
}
