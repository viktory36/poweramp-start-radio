package com.powerampstartradio.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.size
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * A rotary knob composable for controlling a value from -1.0 to +1.0.
 *
 * 270-degree arc from 7 o'clock (–1.0) through 12 o'clock (0.0) to 5 o'clock (+1.0).
 * Drag gesture maps vertical movement to value changes.
 *
 * @param value Current value in [-1.0, 1.0]
 * @param onValueChange Called when the user drags the knob
 * @param modifier Modifier for the knob container
 */
@Composable
fun RotaryKnob(
    value: Float,
    onValueChange: (Float) -> Unit,
    modifier: Modifier = Modifier,
) {
    val trackColor = MaterialTheme.colorScheme.surfaceVariant
    val activeColor = MaterialTheme.colorScheme.primary
    val indicatorColor = MaterialTheme.colorScheme.primary
    val textStyle = MaterialTheme.typography.labelSmall
    val textColor = MaterialTheme.colorScheme.onSurfaceVariant

    // Display: snap to 0.1 increments for display
    val displayValue = (value * 10).roundToInt() / 10f
    val displayText = if (displayValue >= 0f) "+%.1f".format(displayValue) else "%.1f".format(displayValue)

    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        modifier = modifier,
    ) {
        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier
                .size(44.dp)
                .pointerInput(Unit) {
                    detectDragGestures { change, dragAmount ->
                        change.consume()
                        // Drag up = increase, drag down = decrease
                        // Scale: 200px drag = full range
                        val delta = -dragAmount.y / 200f
                        val newValue = (value + delta).coerceIn(-1f, 1f)
                        onValueChange(newValue)
                    }
                },
        ) {
            Canvas(modifier = Modifier.size(40.dp)) {
                val centerX = size.width / 2
                val centerY = size.height / 2
                val radius = size.width / 2 - 4.dp.toPx()
                val strokeWidth = 3.dp.toPx()

                // Arc angles: 270° arc from 135° (7 o'clock) to 45° (5 o'clock)
                // In Canvas: 0° is 3 o'clock, goes clockwise
                val startAngle = 135f  // 7 o'clock
                val sweepAngle = 270f  // to 5 o'clock

                // Background track
                drawArc(
                    color = trackColor,
                    startAngle = startAngle,
                    sweepAngle = sweepAngle,
                    useCenter = false,
                    style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                )

                // Active portion: from center (0 = 12 o'clock = 270°) to current value
                // Value -1 maps to startAngle (135°), +1 maps to startAngle + sweepAngle (405°)
                // 0 maps to 270° (12 o'clock)
                val centerAngle = 270f // 12 o'clock in Canvas coords
                val valueAngle = centerAngle + value * 135f // ±135° from center

                if (abs(value) > 0.01f) {
                    val activeStart = if (value >= 0) centerAngle else valueAngle
                    val activeSweep = abs(value) * 135f
                    drawArc(
                        color = activeColor,
                        startAngle = activeStart,
                        sweepAngle = activeSweep,
                        useCenter = false,
                        style = Stroke(width = strokeWidth, cap = StrokeCap.Round),
                    )
                }

                // Indicator dot at current position
                val indicatorAngle = Math.toRadians(valueAngle.toDouble())
                val dotRadius = 3.dp.toPx()
                drawCircle(
                    color = indicatorColor,
                    radius = dotRadius,
                    center = Offset(
                        centerX + radius * cos(indicatorAngle).toFloat(),
                        centerY + radius * sin(indicatorAngle).toFloat(),
                    ),
                )
            }
        }

        Text(
            text = displayText,
            style = textStyle,
            color = textColor,
        )
    }
}
