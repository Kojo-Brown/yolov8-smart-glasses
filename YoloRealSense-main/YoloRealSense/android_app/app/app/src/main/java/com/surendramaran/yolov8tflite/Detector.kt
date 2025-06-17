package com.surendramaran.yolov8tflite

import android.content.Context
import android.graphics.Bitmap
import android.icu.text.ListFormatter.Width
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import kotlin.math.exp

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    lateinit var imageProcessor: ImageProcessor



    fun setup() {
        val model = FileUtil.loadMappedFile(context, modelPath)
        val options = Interpreter.Options()
        options.numThreads = 4
        interpreter = Interpreter(model, options)

        val inputTensor = interpreter?.getInputTensor(0)
        val outputTensor = interpreter?.getOutputTensor(0)

        if (inputTensor == null || outputTensor == null) {
            Log.e("YOLO", "Input or output tensor is null.")
            return
        }

        val inputShape = interpreter!!.getInputTensor(0).shape()
        val outputShape = interpreter!!.getOutputTensor(0).shape()

        val numChannels = outputShape[2]
        val numClasses = numChannels - 5

        Log.d("YOLO","numClasses = $numClasses")
        Log.d("YOLO", "Input shape: ${inputShape.contentToString()}")
        Log.d("YOLO", "Output shape: ${outputShape.contentToString()}")

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]
        numChannel = outputShape[1] // 21
        numElements = outputShape[2] // =8400
        tensorWidth = inputShape[2]
        tensorHeight = inputShape[1]
        numChannel = outputShape[1] 
        numElements = outputShape[2] 


        imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
            .add(CastOp(INPUT_IMAGE_TYPE))
            .build()
        Log.d("YOLO", "Model loaded: $modelPath")

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line.trim())
                line = reader.readLine()
            }
            Log.d("YOLO", "Loaded labels: ${labels.size} -> $labels")
            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }

    fun clear() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        val originalWidth = frame.width
        val originalHeight = frame.height
        val startTime = System.currentTimeMillis()

        interpreter ?: return
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) return

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)

        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        // VERY IMPORTANT: Check the content of this buffer
        val floatArrayFromBuffer = FloatArray(imageBuffer.capacity() / 4) // Assuming FLOAT32, capacity is total bytes
        imageBuffer.asFloatBuffer().get(floatArrayFromBuffer)
        Log.d("YOLO", "Processed Image Buffer (first 50 values): ${floatArrayFromBuffer.take(50).joinToString()}")
        Log.d("YOLO", "Processed Image Buffer (last 50 values): ${floatArrayFromBuffer.takeLast(50).joinToString()}")
        // Optionally, check a middle section too to ensure it's not just edges that are bad
        Log.d("YOLO", "Processed Image Buffer (middle 50 values): ${floatArrayFromBuffer.slice(floatArrayFromBuffer.size / 2 - 25 until floatArrayFromBuffer.size / 2 + 25).joinToString()}")

        imageBuffer.rewind() // Crucial: Rewind the buffer after reading from it!

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter?.run(imageBuffer, output.buffer)

        val rawOutput = output.floatArray
        // Check first 20 values (which you already have)
        Log.d("YOLO", "First 20 raw output values: ${rawOutput.take(20).joinToString()}")

// Check specific channels for the first few elements
        for (i in 0 until minOf(10, numElements)) { // Log for first 10 potential detections
            val rawCx = rawOutput[i]
            val rawCy = rawOutput[i + numElements]
            val rawW = rawOutput[i + numElements * 2]
            val rawH = rawOutput[i + numElements * 3]
            val rawObjConf = rawOutput[i + numElements * 4]

            var rawMaxClassConf = -Float.MAX_VALUE // Use a very small float for initial comparison
            var rawMaxClassIdx = -1
            for (j in 5 until numChannel) {
                val rawClsConf = rawOutput[i + numElements * j]
                if (rawClsConf > rawMaxClassConf) {
                    rawMaxClassConf = rawClsConf
                    rawMaxClassIdx = j - 5
                }
            }
            Log.d("YOLO", "Raw outputs for element $i: cx=$rawCx, cy=$rawCy, w=$rawW, h=$rawH, objConf=$rawObjConf, maxClsConf=$rawMaxClassConf (idx=$rawMaxClassIdx)")
        }

        val endTime = System.currentTimeMillis()
        val bestBoxes = bestBox(rawOutput, originalWidth, originalHeight)

        if (bestBoxes.isNullOrEmpty()) {
            detectorListener.onEmptyDetect()
        } else {
            detectorListener.onDetect(bestBoxes, endTime - startTime)
        }
    }


    private fun sigmoid(x: Float): Float {
        return (1f / (1f + exp(-x)))
    }

    private fun bestBox(array: FloatArray, frameWidth: Int, frameHeight: Int): List<BoundingBox>? {
        val boxes = mutableListOf<BoundingBox>()

        for (i in 0 until numElements) {
            var maxConf = -1f
            val cx = array[i]
            val cy = array[i + numElements]
            val w  = array[i + numElements * 2]
            val h  = array[i + numElements * 3]
            val objConf = sigmoid(array[i + numElements * 4])

            var maxClassConf = -1f
            var maxClassIdx = -1

            for (j in 4 until (4 + labels.size)) {
                val currentRawScore = array[i + numElements * j]
                if (currentRawScore > maxConf) {
                    maxConf = currentRawScore
                    maxClassIdx = j - 4
                }
            }

            if (maxConf > CONFIDENCE_THRESHOLD) {
                if (maxClassIdx !in labels.indices) {
                    Log.e("YOLO", "Invalid maxClassIdx: $maxClassIdx, skipping detection.")
                    continue
                }

                val label = labels[maxClassIdx]

                // Keep coordinates normalized (0-1) for proper scaling in OverlayView
                val x1 = (cx - w / 2f).coerceIn(0f, 1f)
                val y1 = (cy - h / 2f).coerceIn(0f, 1f)
                val x2 = (cx + w / 2f).coerceIn(0f, 1f)
                val y2 = (cy + h / 2f).coerceIn(0f, 1f)

                boxes.add(
                    BoundingBox(
                        x1 = x1,
                        y1 = y1,
                        x2 = x2,
                        y2 = y2,
                        cx = cx, // keep cx, cy normalized for depth
                        cy = cy, // keep cx, cy normalized for depth
                        w = w,
                        h = h,
                        confidence = maxConf, // Use the raw maxConf directly!
                        cls = maxClassIdx,
                        clsName = label
                    )
                )

                Log.d("YOLO", "Detection confidence (raw): $maxConf, class index: $maxClassIdx, class name: $label")
            }
        }

        return if (boxes.isEmpty()) null else applyNMS(boxes)
    }




    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1_inter = maxOf(box1.x1, box2.x1)
        val y1_inter = maxOf(box1.y1, box2.y1)
        val x2_inter = minOf(box1.x2, box2.x2)
        val y2_inter = minOf(box1.y2, box2.y2)

        val intersectionWidth = x2_inter - x1_inter
        val intersectionHeight = y2_inter - y1_inter

        if (intersectionWidth <= 0 || intersectionHeight <= 0) {
            return 0f
        }

        val intersectionArea = intersectionWidth * intersectionHeight
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

       
        if (box1Area <= 0 || box2Area <= 0) {
            return 0f
        }

        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.20F
        private const val IOU_THRESHOLD = 0.5F
    }
}
