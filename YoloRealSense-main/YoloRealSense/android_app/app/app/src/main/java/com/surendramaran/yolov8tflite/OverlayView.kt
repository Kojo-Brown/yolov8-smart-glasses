package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList
import kotlin.math.max
import kotlin.math.round

//class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
//
//    private var results = listOf<BoundingBox>()
//    private var boxPaint = Paint()
//    private var textBackgroundPaint = Paint()
//
//    private var bounds = Rect()
//    // In your OverlayView class
//    private val textPaint = Paint().apply {
//        color = Color.WHITE
//        textSize = 48f
//        style = Paint.Style.FILL
//        strokeWidth = 2f
//        isAntiAlias = true
//    }
//
//    init {
//        initPaints()
//    }
//    private val distancePaint = Paint().apply {
//        color = Color.CYAN
//        textSize = 40f
//    }
//
//    fun clear() {
//        textPaint.reset()
//        textBackgroundPaint.reset()
//        boxPaint.reset()
//        invalidate()
//        initPaints()
//    }
//
//    private fun initPaints() {
//        textBackgroundPaint.color = Color.BLACK
//        textBackgroundPaint.style = Paint.Style.FILL
//        textBackgroundPaint.textSize = 50f
//
//        textPaint.color = Color.WHITE
//        textPaint.style = Paint.Style.FILL
//        textPaint.textSize = 50f
//
//        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
//        boxPaint.strokeWidth = 8F
//        boxPaint.style = Paint.Style.STROKE
//    }
//
//    override fun draw(canvas: Canvas) {
//        super.draw(canvas)
//
//        results.forEach { box ->
//            // 1. Convert normalized coordinates to view coordinates
//            val left = box.x1 * width
//            val top = box.y1 * height
//            val right = box.x2 * width
//            val bottom = box.y2 * height
//
//            // 2. Draw bounding box (keep your existing style)
//            canvas.drawRect(left, top, right, bottom, boxPaint)
//
//            // 3. Convert distance to step-based description
//            val stepDistance = convertDistanceToStepsShort(box.distance)
//
//            // 4. Prepare combined label text with step-based distance
//            val labelText = "${box.clsName} ($stepDistance)"
//
//            // 5. Calculate text bounds for background
//            textPaint.getTextBounds(labelText, 0, labelText.length, bounds)
//
//            // 6. Draw text background
//            canvas.drawRect(
//                left,
//                top,
//                left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
//                top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
//                textBackgroundPaint
//            )
//
//            // 7. Draw class name + step distance
//            canvas.drawText(labelText, left, top + bounds.height(), textPaint)
//        }
//    }
//
//    private fun convertDistanceToStepsShort(distanceInMeters: Float): String {
//        return when {
//            distanceInMeters <= 0f -> "0s"
//            distanceInMeters < 0.2f -> "close"
//            else -> {
//                val steps = round(distanceInMeters / 0.75f).toInt().coerceAtLeast(1)
//                "${steps}s"
//            }
//        }
//    }
//
//    fun setResults(boundingBoxes: List<BoundingBox>) {
//        results = boundingBoxes
//        invalidate()
//    }
//
//    companion object {
//        private const val BOUNDING_RECT_TEXT_PADDING = 8
//    }
//}

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results = listOf<BoundingBox>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()

    private var bounds = Rect()
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 48f
        style = Paint.Style.FILL
        strokeWidth = 2f
        isAntiAlias = true
    }

    init {
        initPaints()
    }

    private val distancePaint = Paint().apply {
        color = Color.CYAN
        textSize = 40f
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // Only draw objects with valid distance measurements
        results.forEach { box ->
            // Skip objects without valid distance data
            if (box.distance <= 0f) {
                return@forEach
            }

            // 1. Convert normalized coordinates to view coordinates
            val left = box.x1 * width
            val top = box.y1 * height
            val right = box.x2 * width
            val bottom = box.y2 * height

            // 2. Draw bounding box (keep your existing style)
            canvas.drawRect(left, top, right, bottom, boxPaint)

            // 3. Convert distance to step-based description
            val stepDistance = convertDistanceToStepsShort(box.distance)

            // 4. Prepare combined label text with step-based distance
            val labelText = "${box.clsName} ($stepDistance)"

            // 5. Calculate text bounds for background
            textPaint.getTextBounds(labelText, 0, labelText.length, bounds)

            // 6. Draw text background
            canvas.drawRect(
                left,
                top,
                left + bounds.width() + BOUNDING_RECT_TEXT_PADDING,
                top + bounds.height() + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // 7. Draw class name + step distance
            canvas.drawText(labelText, left, top + bounds.height(), textPaint)
        }
    }

    private fun convertDistanceToStepsShort(distanceInMeters: Float): String {
        return when {
            distanceInMeters <= 0f -> "0s"  // This case shouldn't occur now since we filter these out
            distanceInMeters < 0.2f -> "close"
            else -> {
                val steps = round(distanceInMeters / 0.75f).toInt().coerceAtLeast(1)
                "${steps}s"
            }
        }
    }

    fun setResults(boundingBoxes: List<BoundingBox>) {
        // Filter to only include objects with valid distance measurements
        results = boundingBoxes.filter { it.distance > 0f }
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}