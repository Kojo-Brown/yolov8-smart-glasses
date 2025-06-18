package com.surendramaran.yolov8tflite

import android.Manifest
import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognizerIntent
import android.speech.RecognitionListener
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.surendramaran.yolov8tflite.Constants.LABELS_PATH
import com.surendramaran.yolov8tflite.Constants.MODEL_PATH
import com.surendramaran.yolov8tflite.databinding.ActivityMainBinding
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import android.os.Vibrator
import android.os.VibrationEffect
import android.content.Context
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import androidx.annotation.RequiresApi
import com.intel.realsense.librealsense.Config
import com.intel.realsense.librealsense.DepthFrame
import com.intel.realsense.librealsense.DeviceListener
import com.intel.realsense.librealsense.Extension
import com.intel.realsense.librealsense.FrameSet
import com.intel.realsense.librealsense.Pipeline
import com.intel.realsense.librealsense.RsContext
import com.intel.realsense.librealsense.StreamFormat
import com.intel.realsense.librealsense.StreamType
import com.intel.realsense.librealsense.UsbUtilities.ACTION_USB_PERMISSION
import com.intel.realsense.librealsense.VideoFrame
import java.nio.ByteBuffer
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.math.round

enum class DetectionMode {
    START_DETECTION, EXPLAIN_SURROUNDING, READ_MODE
}

data class ObjectPosition(
    val distance: Float,
    val horizontalPosition: String,
    val verticalPosition: String,
    val angle: Float
)

data class EnhancedObjectPosition(
    val distance: Float,
    val detailedHorizontalPosition: String,
    val detailedVerticalPosition: String,
    val angle: Float,
    val angleDescription: String
)

data class SafeRoute(
    val direction: String,
    val description: String,
    val priority: Int
)

data class VoiceCommand(
    val command: String,
    val timestamp: Long = System.currentTimeMillis()
)

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private lateinit var detector: Detector
    private var isDetectionActive = false

    // Thread Management
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var audioExecutor: ExecutorService
    private lateinit var speechExecutor: ExecutorService
    private lateinit var detectionExecutor: ExecutorService

    // Audio Management
    private lateinit var textToSpeech: TextToSpeech
    private var isTTSInitialized = false
    private lateinit var speechRecognizer: SpeechRecognizer
    private lateinit var speechRecognizerIntent: Intent
    private lateinit var audioManager: AudioManager
    private lateinit var audioFocusRequest: AudioFocusRequest
    private lateinit var speechListener: RecognitionListener

    // Thread-safe state management with simplified approach
    private var isListening = false
    private var isProcessingCommand = false
    private val isTTSSpeaking = AtomicBoolean(false)

    // Command queueing (simplified)
    private val commandQueue = ConcurrentLinkedQueue<VoiceCommand>()
    private val ttsCompletionLatch = AtomicReference<CountDownLatch?>()

    // App state
    private lateinit var vibrator: Vibrator
    private var currentMode: DetectionMode = DetectionMode.START_DETECTION
    private val trackedObjects = mutableMapOf<String, Long>()
    private var lastAnnouncementTime = 0L
    private val announcementCooldown = 2000L
    private val minConfidence = 0.25f

    // RealSense
    private lateinit var pipeline: Pipeline
    private var streamingThread: Thread? = null
    private var currentDepthFrame: DepthFrame? = null
    private lateinit var rsContext: RsContext
    private var isPipelineRunning = false
    private var isUsbConnected = false
    private var errorBackoffDelay = 1000L
    private val maxBackoffDelay = 10000L
    // Single frame analysis for explain surrounding
    private var isSingleFrameRequested = false
    private var isAnalyzingFrame = false
    private var analysisStartTime = 0L
    private val ANALYSIS_TIMEOUT = 8000L // 8 seconds

    //For single frame testing
    private var isDetectorBusy = false

    // Text recognition variables
    private val textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private var lastDetectedText = ""
    private var isTextDetectionActive = false


    /**
     * Converts metric distance to intuitive foot step measurements with natural language
     * One step is approximately 0.75 meters
     * Returns integer steps only - no decimals
     */
    private fun convertDistanceToSteps(distanceInMeters: Float): String {
        return when {
            distanceInMeters <= 0f -> "unknown distance"
            distanceInMeters < 0.2f -> "extremely close"
            else -> {
                val steps = round(distanceInMeters / 0.75f).toInt().coerceAtLeast(1)
                if (steps == 1) "1 step away" else "$steps steps away"
            }
        }
    }

    private fun detectTextInBitmap(bitmap: Bitmap) {
        if (!isTextDetectionActive) return

        val image = InputImage.fromBitmap(bitmap, 0)
        textRecognizer.process(image)
            .addOnSuccessListener { visionText ->
                if (visionText.text.isNotEmpty()) {
                    lastDetectedText = visionText.text
                    Log.d(TAG, "Text detected: $lastDetectedText")
                }
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Text detection failed", e)
            }
    }

    private fun readDetectedText() {
        if (lastDetectedText.isNotEmpty()) {
            speak("Reading text: $lastDetectedText")
        } else {
            speak("No text detected yet. Please point camera at text and try again.")
        }
    }

    /**
     * Simple speak method for immediate use
     */
    private fun speak(text: String) {
        if (isTTSInitialized) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    /**
     * Process queued voice commands
     */
    private fun processQueuedCommands() {
        val command = commandQueue.poll()
        command?.let {
            Log.d(TAG, "Processing queued command: ${it.command}")
            // Process the queued command using the same logic
            processVoiceCommandImmediate(it.command)
        }
    }

    /**
     * Process voice command immediately without complex threading
     */
    private fun processVoiceCommandImmediate(spokenText: String) {
        Log.d(TAG, "Processing immediate command: '$spokenText'")

        val command = spokenText.lowercase(Locale.ROOT)

        // If TTS is speaking, queue the command
        if (isTTSSpeaking.get()) {
            Log.d(TAG, "TTS is speaking, queueing command: $command")
            commandQueue.offer(VoiceCommand(command))
            speak("Command received, processing after current announcement")
            return
        }

        // Check if we're already processing a command
        if (isProcessingCommand) {
            Log.d(TAG, "Already processing command, queueing: $command")
            commandQueue.offer(VoiceCommand(command))
            return
        }

        // Process the command based on recognized text
        try {
            isProcessingCommand = true
            var commandRecognized = false

            when {
                command.contains("start detection") -> {
                    Log.d(TAG, "Recognized: START DETECTION")
                    commandRecognized = true
                    handleStartDetectionCommand()
                }

                command.contains("explain surrounding") -> {
                    Log.d(TAG, "Recognized: EXPLAIN SURROUNDING")
                    commandRecognized = true
                    handleExplainSurroundingCommand()
                }

                command.contains("read") -> {
                    Log.d(TAG, "Recognized: READ MODE")
                    commandRecognized = true
                    handleReadModeCommand()
                }

                command.contains("help") -> {
                    Log.d(TAG, "Recognized: HELP")
                    commandRecognized = true
                    handleHelpCommand()
                }

                command.contains("quit") -> {
                    Log.d(TAG, "Recognized: QUIT")
                    commandRecognized = true
                    handleQuitCommand()
                }

                else -> {
                    Log.w(TAG, "Command not recognized: '$command'")
                    speak("Command not recognized. Say 'help' to hear available commands.")
                }
            }

            if (commandRecognized) {
                Log.d(TAG, "Command successfully recognized and processed: $command")
            }

        } catch (e: Exception) {
            Log.e(TAG, "Error processing voice command: $command", e)
            speak("Error processing command. Please try again.")
        } finally {
            // Reset processing flag after a delay to allow command to complete
            Handler(Looper.getMainLooper()).postDelayed({
                isProcessingCommand = false
                Log.d(TAG, "Command processing completed, ready for next command")

                // Process any queued commands
                processQueuedCommands()

                // Resume listening if not in TTS mode
                if (!isTTSSpeaking.get()) {
                    Handler(Looper.getMainLooper()).postDelayed({
                        safeStartListening()
                    }, 500)
                }
            }, 2000) // Give 2 seconds for command processing
        }
    }

    // COMMAND HANDLERS

    private fun handleStartDetectionCommand() {
        Log.d(TAG, "Executing START DETECTION command")

        // Updated message to reflect 10 steps radius
        speak("Start detection activated. Monitoring objects in close proximity within 10 steps radius")
        currentMode = DetectionMode.START_DETECTION
        isTextDetectionActive = false

        if (allPermissionsGranted()) {
            isDetectionActive = true
            trackedObjects.clear()

            val connected = try {
                rsContext.queryDevices().deviceCount > 0
            } catch (e: Exception) {
                Log.e(TAG, "Failed to query RealSense devices", e)
                false
            }

            if (connected) {
                Log.d(TAG, "RealSense device detected — checking permissions")
                checkUsbPermissions()
            } else {
                speak("RealSense camera is not connected")
                Log.w(TAG, "RealSense not connected according to rsContext")
            }
        }
    }

    private fun handleExplainSurroundingCommand() {
        Log.d(TAG, "Executing EXPLAIN SURROUNDING command")

        // Reset any previous analysis state
        isSingleFrameRequested = false
        isAnalyzingFrame = false

        currentMode = DetectionMode.EXPLAIN_SURROUNDING
        isTextDetectionActive = false

        if (allPermissionsGranted()) {
            val connected = try {
                rsContext.queryDevices().deviceCount > 0
            } catch (e: Exception) {
                Log.e(TAG, "Failed to query RealSense devices", e)
                false
            }

            if (connected) {
                Log.d(TAG, "RealSense device detected for single frame analysis")

                // Start the pipeline if not already running
                if (!isPipelineRunning) {
                    speak("Starting camera for analysis...")
                    checkUsbPermissions()
                    // Give time for pipeline to start
                    Handler(Looper.getMainLooper()).postDelayed({
                        if (currentMode == DetectionMode.EXPLAIN_SURROUNDING) {
                            requestSingleFrameAnalysis()
                        }
                    }, 2000)
                } else {
                    requestSingleFrameAnalysis()
                }
            } else {
                speak("RealSense camera is not connected")
                Log.w(TAG, "RealSense not connected")
            }
        } else {
            speak("Camera permission required for analysis")
        }
    }

    private fun requestSingleFrameAnalysis() {
        if (isAnalyzingFrame) {
            Log.w(TAG, "Analysis already in progress, ignoring request")
            return
        }

        speak("Analyzing current view. Please hold camera steady.")

        // Set analysis state
        isSingleFrameRequested = true
        isAnalyzingFrame = true
        analysisStartTime = System.currentTimeMillis()

        Log.d(TAG, "Single frame analysis requested at ${analysisStartTime}")

        // Set timeout to prevent freezing
        Handler(Looper.getMainLooper()).postDelayed({
            if (isAnalyzingFrame && (System.currentTimeMillis() - analysisStartTime) > ANALYSIS_TIMEOUT) {
                Log.w(TAG, "Single frame analysis timeout - resetting state")
                resetAnalysisState()
                speak("Analysis timeout. Please try again.")
            }
        }, ANALYSIS_TIMEOUT)
    }

    private fun resetAnalysisState() {
        isSingleFrameRequested = false
        isAnalyzingFrame = false
        analysisStartTime = 0L
        Log.d(TAG, "Analysis state reset")
    }

    private fun handleReadModeCommand() {
        Log.d(TAG, "Executing READ MODE command")

        speak("Read mode activated. Scanning for text.")
        currentMode = DetectionMode.READ_MODE
        isTextDetectionActive = true
        lastDetectedText = ""

        if (allPermissionsGranted()) {
            isDetectionActive = true

            val connected = try {
                rsContext.queryDevices().deviceCount > 0
            } catch (e: Exception) {
                Log.e(TAG, "Failed to query RealSense devices", e)
                false
            }

            if (connected) {
                checkUsbPermissions()
            } else {
                speak("RealSense camera is not connected")
            }
        }

        // After a brief delay, start reading detected text
        Handler(Looper.getMainLooper()).postDelayed({
            readDetectedText()
        }, 3000)
    }

    private fun handleHelpCommand() {
        Log.d(TAG, "Executing HELP command")

        val helpMessage = "Available commands: " +
                "Start detection - for monitoring nearby objects within 3 steps. " +
                "Explain surrounding - for detailed environmental scanning at all distances. " +
                "Read - for text recognition and reading. " +
                "Help - to hear this message again. " +
                "Quit - to exit the application."

        speak(helpMessage)
    }

    private fun handleQuitCommand() {
        Log.d(TAG, "Executing QUIT command")

        speak("Goodbye. Closing application.")

        // Stop detection and camera
        isDetectionActive = false
        isTextDetectionActive = false
        stopCamera()

        // Close the app after TTS completes
        Handler(Looper.getMainLooper()).postDelayed({
            finishAffinity()
        }, 2000)
    }

    // OBJECT DETECTION HELPER METHODS

    private fun getObjectPosition(box: BoundingBox, frameWidth: Int, frameHeight: Int): ObjectPosition {
        val centerX = (box.x1 + box.x2) / 2f
        val centerY = (box.y1 + box.y2) / 2f

        val horizontalPosition = when {
            centerX < 0.33f -> "left"
            centerX > 0.67f -> "right"
            else -> "center"
        }

        val verticalPosition = when {
            centerY < 0.33f -> "upper"
            centerY > 0.67f -> "lower"
            else -> "middle"
        }

        // Calculate angle from center (0 degrees = straight ahead)
        val angle = ((centerX - 0.5f) * 60f) // Assuming 60-degree field of view

        return ObjectPosition(box.distance, horizontalPosition, verticalPosition, angle)
    }

    // Enhanced object position analysis (more robust)
    private fun getEnhancedObjectPosition(box: BoundingBox, frameWidth: Int, frameHeight: Int): EnhancedObjectPosition {
        val centerX = (box.x1 + box.x2) / 2f
        val centerY = (box.y1 + box.y2) / 2f

        // More detailed horizontal positioning
        val detailedHorizontalPosition = when {
            centerX < 0.2f -> "far left"
            centerX < 0.35f -> "left"
            centerX < 0.45f -> "center-left"
            centerX < 0.55f -> "center"
            centerX < 0.65f -> "center-right"
            centerX < 0.8f -> "right"
            else -> "far right"
        }

        // More detailed vertical positioning
        val detailedVerticalPosition = when {
            centerY < 0.25f -> "high"
            centerY < 0.4f -> "upper"
            centerY < 0.6f -> "middle"
            centerY < 0.75f -> "lower"
            else -> "ground level"
        }

        // Calculate angle from center with more precision
        val angle = ((centerX - 0.5f) * 60f) // Assuming 60-degree field of view
        val angleDescription = when {
            angle < -30f -> "sharp left"
            angle < -15f -> "left"
            angle < -5f -> "slightly left"
            angle <= 5f -> "straight ahead"
            angle <= 15f -> "slightly right"
            angle <= 30f -> "right"
            else -> "sharp right"
        }

        return EnhancedObjectPosition(
            distance = box.distance,
            detailedHorizontalPosition = detailedHorizontalPosition,
            detailedVerticalPosition = detailedVerticalPosition,
            angle = angle,
            angleDescription = angleDescription
        )
    }

    private fun prioritizeObjects(objectsWithPositions: List<Pair<BoundingBox, ObjectPosition>>): List<Pair<BoundingBox, ObjectPosition>> {
        return objectsWithPositions.sortedWith(compareBy<Pair<BoundingBox, ObjectPosition>> {
            // Priority 1: Distance (closer = higher priority)
            it.second.distance
        }.thenBy {
            // Priority 2: Center objects get higher priority
            when(it.second.horizontalPosition) {
                "center" -> 0
                "left", "right" -> 1
                else -> 2
            }
        }.thenByDescending {
            // Priority 3: Higher confidence
            it.first.confidence
        })
    }


    private fun getDetailedObjectName(clsName: String): String {
        return when(clsName.lowercase()) {
            "bicycle" -> "bicycle"
            "bus" -> "bus"
            "car" -> "car"
            "dog" -> "dog"
            "electric pole" -> "electric pole"
            "motorcycle" -> "motorcycle"
            "person" -> "person"
            "traffic signs" -> "traffic sign"
            "tree" -> "tree"
            "uncovered manhole" -> "uncovered manhole"
            "green pedestrian light" -> "green pedestrian light"
            "no pedestrian crossing sign" -> "no pedestrian crossing sign"
            "no pedestrian lane" -> "no pedestrian lane"
            "pedestrian crossing sign" -> "pedestrian crossing sign"
            "pedestrian lane" -> "pedestrian lane"
            "red pedestrian light" -> "red pedestrian light"
            else -> clsName
        }
    }

//    private fun handleStartDetection(detectedBoxes: List<BoundingBox>, currentTime: Long) {
//        // Filter objects within approximately 10 steps (7.5 meters) for start detection monitoring
//        val nearbyObjects = detectedBoxes.filter { it.distance <= 7.5f && it.distance > 0f }
//
//        if (nearbyObjects.isEmpty() || textToSpeech.isSpeaking ||
//            currentTime - lastAnnouncementTime < announcementCooldown) {
//            return
//        }
//
//        val frameWidth = currentDepthFrame?.width ?: 640
//        val frameHeight = currentDepthFrame?.height ?: 480
//
//        // Get positions for all nearby objects
//        val objectsWithPositions = nearbyObjects.map { box ->
//            val position = getObjectPosition(box, frameWidth, frameHeight)
//            Pair(box, position)
//        }
//
//        // Prioritize objects by distance and position
//        val prioritizedObjects = prioritizeObjects(objectsWithPositions)
//
//        // Take top 3 most important objects for concise announcement
//        val topObjects = prioritizedObjects.take(3)
//
//        if (topObjects.isNotEmpty()) {
//            // Generate safe route suggestions
//            val safeRoutes = generateSafeRoutes(objectsWithPositions)
//
//            // Build concise announcement with integrated navigation
//            val message = buildConciseDetectionMessage(topObjects, safeRoutes)
//
//            // Use vibration patterns based on proximity
//            val closestDistance = topObjects.minOf { it.first.distance }
//            applyVibrationPattern(closestDistance)
//
//            speak(message)
//            lastAnnouncementTime = currentTime
//        }
//    }

    private fun handleStartDetection(detectedBoxes: List<BoundingBox>, currentTime: Long) {
        // All boxes passed here already have valid distances (> 0f)
        // Filter objects within approximately 10 steps (7.5 meters) for start detection monitoring
        val nearbyObjects = detectedBoxes.filter { it.distance <= 7.5f }

        if (nearbyObjects.isEmpty() || textToSpeech.isSpeaking ||
            currentTime - lastAnnouncementTime < announcementCooldown) {
            return
        }

        val frameWidth = currentDepthFrame?.width ?: 640
        val frameHeight = currentDepthFrame?.height ?: 480

        // Get positions for all nearby objects (all have valid distances)
        val objectsWithPositions = nearbyObjects.map { box ->
            val position = getObjectPosition(box, frameWidth, frameHeight)
            Pair(box, position)
        }

        // Prioritize objects by distance and position
        val prioritizedObjects = prioritizeObjects(objectsWithPositions)

        // Take top 3 most important objects for concise announcement
        val topObjects = prioritizedObjects.take(3)

        if (topObjects.isNotEmpty()) {
            // Generate safe route suggestions
            val safeRoutes = generateSafeRoutes(objectsWithPositions)

            // Build concise announcement with integrated navigation
            val message = buildConciseDetectionMessage(topObjects, safeRoutes)

            // Use vibration patterns based on proximity
            val closestDistance = topObjects.minOf { it.first.distance }
            applyVibrationPattern(closestDistance)

            speak(message)
            lastAnnouncementTime = currentTime
        }
    }

    private fun buildConciseDetectionMessage(
        objects: List<Pair<BoundingBox, ObjectPosition>>,
        safeRoutes: List<SafeRoute>
    ): String {
        val closestObject = objects.first() // Already prioritized
        val (box, position) = closestObject

        val objectName = getSpecificObjectName(box.clsName)
        val distance = convertDistanceToSteps(box.distance)
        val direction = position.horizontalPosition

        return when {
            // Very close objects - immediate safety alert
            box.distance < 0.75f -> {
                val route = if (safeRoutes.isNotEmpty()) {
                    when (safeRoutes.first().direction) {
                        "left" -> "step left"
                        "right" -> "step right"
                        "stop" -> "stop"
                        else -> "be careful"
                    }
                } else "be careful"

                "Warning: $objectName $distance ahead $direction, $route"
            }

            // Close objects with multiple detections
            objects.size > 1 && box.distance < 2.25f -> {
                val secondObject = objects[1]
                val secondName = getSpecificObjectName(secondObject.first.clsName)
                val secondDistance = convertDistanceToSteps(secondObject.first.distance)
                val secondDirection = secondObject.second.horizontalPosition

                val route = if (safeRoutes.isNotEmpty()) {
                    when (safeRoutes.first().direction) {
                        "left" -> "go left"
                        "right" -> "go right"
                        "forward" -> "continue straight"
                        else -> "proceed carefully"
                    }
                } else "proceed carefully"

                "$objectName $distance $direction, $secondName $secondDistance $secondDirection - $route"
            }

            // Single close object
            box.distance < 2.25f -> {
                val route = if (safeRoutes.isNotEmpty()) {
                    when (safeRoutes.first().direction) {
                        "left" -> "keep left"
                        "right" -> "keep right"
                        "forward" -> "continue ahead"
                        else -> "proceed carefully"
                    }
                } else "noted"

                "$objectName $distance $direction - $route"
            }

            // Distant monitoring (3-10 steps)
            else -> {
                val additionalCount = objects.size - 1
                val baseMessage = "$objectName $distance $direction"

                if (additionalCount > 0) {
                    "$baseMessage plus $additionalCount more objects ahead"
                } else {
                    "$baseMessage ahead"
                }
            }
        }
    }

    private fun applyVibrationPattern(closestDistance: Float) {
        when {
            closestDistance < 0.5f -> { // Less than half a step
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createWaveform(longArrayOf(0, 200, 100, 200), -1))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(longArrayOf(0, 200, 100, 200), -1)
                }
            }
            closestDistance < 1.0f -> { // Within one step
                safeVibrate()
            }
            closestDistance < 3.0f -> { // Within 3 steps
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(50)
                }
            }
            else -> { // Beyond 3 steps but within 10 steps
                if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                    vibrator.vibrate(VibrationEffect.createOneShot(20, VibrationEffect.DEFAULT_AMPLITUDE))
                } else {
                    @Suppress("DEPRECATION")
                    vibrator.vibrate(20)
                }
            }
        }
    }

    private fun generateSafeRoutes(objectsWithPositions: List<Pair<BoundingBox, ObjectPosition>>): List<SafeRoute> {
        val routes = mutableListOf<SafeRoute>()

        // Check for obstacles in different directions using step-based measurements
        val leftObstacles = objectsWithPositions.filter {
            it.second.horizontalPosition == "left" && it.second.distance < 2.25f // Within 3 steps
        }
        val centerObstacles = objectsWithPositions.filter {
            it.second.horizontalPosition == "center" && it.second.distance < 2.25f // Within 3 steps
        }
        val rightObstacles = objectsWithPositions.filter {
            it.second.horizontalPosition == "right" && it.second.distance < 2.25f // Within 3 steps
        }

        // Suggest safest routes based on obstacle distribution with detailed descriptions
        when {
            centerObstacles.isEmpty() && leftObstacles.isEmpty() && rightObstacles.isEmpty() -> {
                routes.add(SafeRoute("forward", "Clear path ahead for at least 3 steps. Continue straight with confidence", 1))
            }
            centerObstacles.isNotEmpty() -> {
                val centerDistance = centerObstacles.minOf { it.second.distance }
                val centerObject = centerObstacles.minByOrNull { it.second.distance }
                val objectName = centerObject?.let { getSpecificObjectName(it.first.clsName) } ?: "obstacle"
                val distanceDescription = convertDistanceToSteps(centerDistance)

                when {
                    leftObstacles.isEmpty() && rightObstacles.isNotEmpty() -> {
                        routes.add(SafeRoute("left", "Move to your left to bypass $objectName $distanceDescription ahead. Left side is clear", 1))
                    }
                    rightObstacles.isEmpty() && leftObstacles.isNotEmpty() -> {
                        routes.add(SafeRoute("right", "Move to your right to avoid $objectName $distanceDescription ahead. Right side is clear", 1))
                    }
                    leftObstacles.isEmpty() && rightObstacles.isEmpty() -> {
                        routes.add(SafeRoute("left", "Step to your left to bypass $objectName $distanceDescription ahead", 1))
                        routes.add(SafeRoute("right", "Step to your right to bypass $objectName $distanceDescription ahead", 2))
                    }
                    else -> {
                        val totalObstacles = leftObstacles.size + centerObstacles.size + rightObstacles.size
                        routes.add(SafeRoute("stop", "$totalObstacles obstacles detected within 3 steps. Stop and reassess your surroundings", 1))
                    }
                }
            }
            leftObstacles.isNotEmpty() && rightObstacles.isEmpty() -> {
                val leftDistance = leftObstacles.minOf { it.second.distance }
                val leftObject = leftObstacles.minByOrNull { it.second.distance }
                val objectName = leftObject?.let { getSpecificObjectName(it.first.clsName) } ?: "obstacle"
                val distanceDescription = convertDistanceToSteps(leftDistance)
                routes.add(SafeRoute("right", "Bear right to avoid $objectName $distanceDescription on your left. Right path is clear", 1))
            }
            rightObstacles.isNotEmpty() && leftObstacles.isEmpty() -> {
                val rightDistance = rightObstacles.minOf { it.second.distance }
                val rightObject = rightObstacles.minByOrNull { it.second.distance }
                val objectName = rightObject?.let { getSpecificObjectName(it.first.clsName) } ?: "obstacle"
                val distanceDescription = convertDistanceToSteps(rightDistance)
                routes.add(SafeRoute("left", "Bear left to avoid $objectName $distanceDescription on your right. Left path is clear", 1))
            }
            else -> {
                // Multiple obstacles on both sides
                val closestObstacle = objectsWithPositions.minByOrNull { it.second.distance }
                val closestDistance = closestObstacle?.second?.distance ?: 0f
                when {
                    closestDistance < 0.75f -> {
                        routes.add(SafeRoute("stop", "Multiple obstacles very close. Stop immediately and reassess", 1))
                    }
                    leftObstacles.size < rightObstacles.size -> {
                        routes.add(SafeRoute("left", "Multiple obstacles detected. Left side has fewer obstacles. Proceed carefully left", 1))
                    }
                    rightObstacles.size < leftObstacles.size -> {
                        routes.add(SafeRoute("right", "Multiple obstacles detected. Right side has fewer obstacles. Proceed carefully right", 1))
                    }
                    else -> {
                        routes.add(SafeRoute("careful", "Complex obstacle pattern detected. Proceed very slowly straight ahead", 1))
                    }
                }
            }
        }

        return routes.sortedBy { it.priority }
    }

    private fun getSpecificObjectName(clsName: String): String {
        return when(clsName.lowercase().trim()) {
            "bicycle" -> "bicycle"
            "bus" -> "bus"
            "car" -> "car"
            "dog" -> "dog"
            "electric pole" -> "electric pole"
            "motorcycle" -> "motorcycle"
            "person" -> "person"
            "traffic signs" -> "traffic sign"
            "tree" -> "tree"
            "uncovered manhole" -> "manhole"
            "green pedestrian light" -> "green light"
            "no pedestrian crossing sign" -> "no crossing sign"
            "no pedestrian lane" -> "no pedestrian lane"
            "pedestrian crossing sign" -> "crossing sign"
            "pedestrian lane" -> "pedestrian lane"
            "red pedestrian light" -> "red light"
            else -> clsName
        }
    }

    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                ACTION_USB_PERMISSION -> {

                    val device = intent.getParcelableExtra<UsbDevice>(UsbManager.EXTRA_DEVICE)
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {

                        Log.d(TAG, "USB permission granted for ${device?.deviceName}")
                        isUsbConnected = true

                        configureAudioRouting()
                        startRealsensePipeline()


                    } else {
                        Log.w(TAG, "USB permission denied for ${device?.deviceName}")
                    }
                }
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    Log.d(TAG, "USB attached")

                    isUsbConnected = true
                    configureAudioRouting()
                    pauseSpeechRecognition()


                    val count = try {
                        rsContext.queryDevices().deviceCount
                    } catch (e: Exception) {
                        Log.e(TAG, "Error querying RealSense devices in USB attach", e)
                        0
                    }

                    Log.d(TAG, "Device count at attach time: $count")

                    if (count > 0) {
                        Log.d(TAG, "Device present, checking permissions")
                        checkUsbPermissions()
                    } else {
                        Log.w(TAG, "No devices found at USB attach time — waiting 500ms")
                        Handler(Looper.getMainLooper()).postDelayed({
                            val retryCount = rsContext.queryDevices().deviceCount
                            Log.d(TAG, "Device count after delay: $retryCount")
                            if (retryCount > 0) checkUsbPermissions()
                            else Log.e(TAG, "Still no RealSense devices found")
                        }, 500)
                    }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    Log.d(TAG, "USB disconnected - restoring audio")
                    isUsbConnected = false
                    resumeSpeechRecognition()
                }
            }
        }
    }

    private fun configureAudioRouting() {
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.isBluetoothScoOn = false
        audioManager.isSpeakerphoneOn = true
        audioManager.setParameters("no_usb_audio=1")
        audioManager.mode = AudioManager.MODE_IN_COMMUNICATION
        audioManager.setParameters("no_usb_audio=1")
    }

    private fun pauseSpeechRecognition() {
        stopListening()
    }

    private fun resumeSpeechRecognition() {
        if (!isProcessingCommand && !isUsbConnected) {
            safeStartListening()
        }
    }

    private fun checkUsbPermissions() {
        val usbManager = getSystemService(Context.USB_SERVICE) as UsbManager
        val deviceList = usbManager.deviceList.values.firstOrNull()

        Log.d(TAG, "USB device list size: ${usbManager.deviceList.size}")
        Log.d(TAG, "RsContext device count: ${rsContext.queryDevices().deviceCount}")

        deviceList?.let { device ->
            if (!usbManager.hasPermission(device)) {
                val permissionIntent = PendingIntent.getBroadcast(
                    this,
                    0,
                    Intent(ACTION_USB_PERMISSION),
                    PendingIntent.FLAG_IMMUTABLE
                )
                usbManager.requestPermission(device, permissionIntent)

            } else {
                Log.d(TAG, "USB permission already granted for ${device.deviceName}")
                isUsbConnected = true
                configureAudioRouting()
                startRealsensePipeline()
            }
        }
    }

    private fun startRealsensePipeline(): Boolean {
        return try {
            pipeline = Pipeline().apply {
                start(Config().apply {
                    enableStream(StreamType.COLOR, 640, 480, StreamFormat.RGBA8)
                    enableStream(StreamType.DEPTH, 640, 480, StreamFormat.Z16)
                })
                Log.d(TAG, "Pipeline started with profile")
            }

            streamingThread = Thread {
                while (!Thread.interrupted()) {
                    try {
                        pipeline.waitForFrames(5000)?.use { frames ->
                            processFrame(frames)
                        }
                    } catch (e: RuntimeException) {
                        if (e.message?.contains("timeout") == true) {
                            Log.w(TAG, "Frame timeout - retrying...")
                        } else {
                            Log.e(TAG, "Frame processing error", e)
                        }
                    }
                    catch (e: InterruptedException) {
                        Log.e("YOLO", "Realsense thread interrupted during sleep")
                        break
                    }
                }
            }.apply { start() }

            isPipelineRunning = true
            true

        } catch (e: Exception) {
            Log.e(TAG, "Pipeline startup failed", e)
            false
        }
    }

    private fun stopRealsensePipeline() {
        streamingThread?.let { thread ->
            thread.interrupt()
            try {
                thread.join(1000)

            } catch (e: InterruptedException) {
                Log.w(TAG, "Thread interruption warning", e)
            }
        }

        streamingThread = null

        if (::pipeline.isInitialized && isPipelineRunning) {
            try {
                pipeline.stop()
                isPipelineRunning = false
                Log.d(TAG, "Pipeline stopped successfully")

            } catch (e: Exception) {
                Log.e(TAG, "Pipeline stop failed", e)
            }
        }
    }

    private fun processFrame(frames: FrameSet) {
        frames.first(StreamType.COLOR)?.use { colorFrame ->
            if (colorFrame.`is`(Extension.VIDEO_FRAME)) {
                val videoFrame = colorFrame.`as`<VideoFrame>(Extension.VIDEO_FRAME)
                val bitmap = videoFrame.toBitmap()

                frames.first(StreamType.DEPTH)?.use { depthFrame ->
                    if (depthFrame.`is`(Extension.DEPTH_FRAME)) {
                        currentDepthFrame = depthFrame.`as`(Extension.DEPTH_FRAME)
                    }
                }

                runOnUiThread {
                    binding.cameraPreview.setImageBitmap(bitmap)
                    binding.inferenceTime.text = ""
                }

                // Text detection for READ_MODE
                if (currentMode == DetectionMode.READ_MODE && isTextDetectionActive) {
                    detectTextInBitmap(bitmap)
                }

                // IMPROVED Single frame analysis for EXPLAIN_SURROUNDING
                if (currentMode == DetectionMode.EXPLAIN_SURROUNDING &&
                    isSingleFrameRequested &&
                    isAnalyzingFrame &&
                    !isDetectorBusy) {

                    Log.d(TAG, "Processing single frame for explain surrounding")

                    // Don't reset the flag immediately - wait for detection to complete
                    if (::detector.isInitialized) {
                        isDetectorBusy = true
                        detectionExecutor.execute {
                            try {
                                Log.d(TAG, "Starting detector.detect() for single frame")
                                val startTime = System.currentTimeMillis()
                                detector.detect(bitmap)
                                val endTime = System.currentTimeMillis()
                                Log.d(TAG, "Single frame detection completed in ${endTime - startTime}ms")

                                // Note: Don't reset state here - let onDetect handle it
                            } catch (e: Exception) {
                                Log.e(TAG, "Error in single frame detection", e)
                                isDetectorBusy = false
                                runOnUiThread {
                                    speak("Analysis failed. Please try again.")
                                    resetAnalysisState()
                                }
                            }
                        }
                    } else {
                        Log.e(TAG, "Detector not initialized")
                        speak("Detection system not ready. Please try again.")
                        resetAnalysisState()
                    }
                }

                // Continuous detection for START_DETECTION mode only
                if (currentMode == DetectionMode.START_DETECTION &&
                    isDetectionActive &&
                    ::detector.isInitialized &&
                    !isDetectorBusy &&
                    !isAnalyzingFrame) { // Don't interfere with single frame analysis

                    isDetectorBusy = true
                    detectionExecutor.execute {
                        try {
                            val startTime = System.currentTimeMillis()
                            detector.detect(bitmap)
                            val endTime = System.currentTimeMillis()
                            Log.d(TAG, "Continuous detection took ${endTime - startTime}ms")
                            isDetectorBusy = false
                        } catch (e: Exception) {
                            Log.e(TAG, "Error in continuous detection", e)
                            isDetectorBusy = false
                        }
                    }
                }
            }
        }
    }

    private fun VideoFrame.toBitmap(): Bitmap {
        return try {
            val byteArray = ByteArray(dataSize)
            getData(byteArray)
            val buffer = ByteBuffer.wrap(byteArray)

            val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
            bmp.copyPixelsFromBuffer(buffer)

            bmp
        } catch (e: Exception) {
            Log.e(TAG, "Bitmap conversion failed", e)
            Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        }
    }

    /**
     * Start speech recognition with proper error handling and logging
     */
    private fun safeStartListening() {
        if (!isProcessingCommand && acquireAudioFocus()) {
            try {
                Log.d(TAG, "Attempting to start listening...")

                // Check permissions
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                    != PackageManager.PERMISSION_GRANTED) {
                    Log.e(TAG, "Audio permission not granted")
                    requestAudioPermission()
                    return
                }

                // Check if speech recognition is available
                if (!SpeechRecognizer.isRecognitionAvailable(this)) {
                    Log.e(TAG, "Speech recognition not available on this device")
                    return
                }

                speechRecognizer.startListening(speechRecognizerIntent)
                isListening = true
                Log.d(TAG, "Speech recognition started successfully")
            } catch (e: SecurityException) {
                Log.e(TAG, "Microphone access denied", e)
                requestAudioPermission()
            } catch (e: IllegalStateException) {
                Log.e(TAG, "Recognizer in bad state", e)
                resetRecognizer()
            } catch (e: Exception) {
                Log.e(TAG, "Error starting speech recognition", e)
                scheduleRetryWithBackoff()
            }
        } else {
            Log.d(TAG, "Cannot start listening - processing: $isProcessingCommand, audio focus available")
        }
    }

    private fun stopListening() {
        if (isListening) {
            try {
                speechRecognizer.stopListening()
                isListening = false
                Log.d(TAG, "SpeechRecognizer: stopped listening")
            } catch (e: Exception) {
                Log.e(TAG, "Speech recognition stop failed", e)
            }
        }
    }

    private fun resetRecognizer() {
        stopListening()
        isListening = false

        try {
            speechRecognizer.destroy()
        } catch (e: Exception) {
            Log.w(TAG, "Failed to destroy speech recognizer", e)
        }

        Handler(Looper.getMainLooper()).postDelayed({
            if (SpeechRecognizer.isRecognitionAvailable(this)) {
                speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
                speechRecognizer.setRecognitionListener(speechListener)
                Log.d(TAG, "SpeechRecognizer reset complete")
                Handler(Looper.getMainLooper()).postDelayed({
                    safeStartListening()
                }, 300)
            } else {
                Log.e(TAG, "Speech recognition not available on device")
            }
        }, 300)
    }

    private fun scheduleRetryWithBackoff() {
        Handler(Looper.getMainLooper()).postDelayed({
            if (!isProcessingCommand) {
                safeStartListening()
            }
        }, errorBackoffDelay)
        errorBackoffDelay = (errorBackoffDelay * 2).coerceAtMost(maxBackoffDelay)
    }

    private fun acquireAudioFocus(): Boolean {
        return audioManager.requestAudioFocus(audioFocusRequest) ==
                AudioManager.AUDIOFOCUS_REQUEST_GRANTED
    }

    private fun requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_CODE_AUDIO_PERMISSION
            )
        }
    }

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val filter = IntentFilter().apply {
            addAction(ACTION_USB_PERMISSION)
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }

        registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        audioExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "AudioThread").apply {
                priority = Thread.MAX_PRIORITY
            }
        }
        speechExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "SpeechThread").apply {
                priority = Thread.MAX_PRIORITY
            }
        }
        cameraExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "CameraThread")
        }
        detectionExecutor = Executors.newSingleThreadExecutor { r ->
            Thread(r, "DetectionThread")
        }

        audioManager = getSystemService(AUDIO_SERVICE) as AudioManager

        audioFocusRequest = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT)
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_ACCESSIBILITY)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
            .build()

        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        detector = Detector(baseContext, MODEL_PATH, LABELS_PATH, this)
        detector.setup()

        RsContext.init(applicationContext)

        rsContext = RsContext().apply {
            setDevicesChangedCallback(object : DeviceListener {
                override fun onDeviceAttach() {
                    Log.d(TAG, "RealSense device connected (onCreate)")
                    isUsbConnected = true
                }

                override fun onDeviceDetach() {
                    Log.d(TAG, "RealSense device disconnected (onCreate)")
                    isUsbConnected = false
                }
            })
        }

        Log.d(TAG, "Initial RealSense device count: ${rsContext.queryDevices().deviceCount}")

        Handler(Looper.getMainLooper()).postDelayed({
            textToSpeech = TextToSpeech(this) { status ->
                if (status == TextToSpeech.SUCCESS) {
                    val result = textToSpeech.setLanguage(Locale.US)
                    isTTSInitialized = result != TextToSpeech.LANG_MISSING_DATA &&
                            result != TextToSpeech.LANG_NOT_SUPPORTED

                    if (!isTTSInitialized) {
                        Log.e(TAG, "TTS Language not supported")
                    } else {
                        // Set TTS completion listener
                        textToSpeech.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                            override fun onStart(utteranceId: String?) {
                                Log.d(TAG, "TTS started: $utteranceId")
                            }

                            override fun onDone(utteranceId: String?) {
                                Log.d(TAG, "TTS completed: $utteranceId")
                                ttsCompletionLatch.get()?.countDown()
                            }

                            override fun onError(utteranceId: String?) {
                                Log.e(TAG, "TTS error: $utteranceId")
                                ttsCompletionLatch.get()?.countDown()
                            }
                        })
                    }
                } else {
                    Log.e(TAG, "TTS Initialization failed")
                }
            }
        }, 1000)

        // Setup Speech Recognition
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
        speechRecognizerIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
            putExtra(RecognizerIntent.EXTRA_AUDIO_SOURCE, MediaRecorder.AudioSource.MIC)
        }

        speechListener = object : RecognitionListener {
            override fun onResults(results: Bundle?) {
                val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.let { resultList ->
                    if (resultList.isNotEmpty()) {
                        val spokenText = resultList[0].lowercase(Locale.ROOT).trim()
                        Log.d(TAG, "Recognized speech: '$spokenText'")

                        if (!isProcessingCommand) {
                            isProcessingCommand = true

                            if (textToSpeech.isSpeaking) {
                                textToSpeech.stop()
                            }

                            when {
                                spokenText.contains("start") || spokenText.contains("detection") -> {
                                    Log.d(TAG, "Processing: START DETECTION")
                                    handleStartDetectionCommand()
                                }

                                spokenText.contains("explain") || spokenText.contains("surrounding") -> {
                                    Log.d(TAG, "Processing: EXPLAIN SURROUNDING")
                                    handleExplainSurroundingCommand()
                                }

                                spokenText.contains("read") || spokenText.contains("text") -> {
                                    Log.d(TAG, "Processing: READ MODE")
                                    handleReadModeCommand()
                                }

                                spokenText.contains("help") -> {
                                    Log.d(TAG, "Processing: HELP")
                                    handleHelpCommand()
                                }

                                spokenText.contains("quit") -> {
                                    Log.d(TAG, "Processing: QUIT")
                                    handleQuitCommand()
                                }

                                // Ignore very short or unclear speech
                                spokenText.length < 3 -> {
                                    Log.d(TAG, "Speech too short, ignoring: '$spokenText'")
                                }

                                else -> {
                                    // Only announce error for longer, clear speech
                                    if (spokenText.length > 4) {
                                        Log.w(TAG, "Command not recognized: '$spokenText'")
                                        speak("Command not recognized.")
                                    }
                                }
                            }

                            Handler(Looper.getMainLooper()).postDelayed({
                                isProcessingCommand = false
                                safeStartListening()
                            }, 1500)
                        } else {
                            Log.d(TAG, "Already processing command, ignoring: '$spokenText'")
                        }
                    }
                }
                errorBackoffDelay = 1000L
            }

            override fun onError(error: Int) {
                isListening = false
                Log.e(TAG, "Speech recognition error: $error")

                when (error) {
                    SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> {
                        Log.e(TAG, "Missing permissions")
                        requestAudioPermission()
                    }
                    SpeechRecognizer.ERROR_CLIENT -> {
                        resetRecognizer()
                    }
                    SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> {
                        scheduleRetryWithBackoff()
                    }
                    SpeechRecognizer.ERROR_NO_MATCH -> {
                        Log.w(TAG, "No match - resuming listening")
                        scheduleRetryWithBackoff()
                    }
                    else -> {
                        scheduleRetryWithBackoff()
                    }
                }
            }

            override fun onReadyForSpeech(params: Bundle?) {
                Log.d(TAG, "Ready for speech - you should hear a beep now")
            }

            override fun onBeginningOfSpeech() {
                Log.d(TAG, "Speech started")
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}

            override fun onEndOfSpeech() {
                Log.d(TAG, "End of speech detected")
                isListening = false
                if (!isProcessingCommand) {
                    scheduleRetryWithBackoff()
                }
            }

            override fun onPartialResults(partialResults: Bundle?) {
                val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                matches?.let {
                    if (it.isNotEmpty()) {
                        Log.d(TAG, "Partial result: ${it[0]}")
                    }
                }
            }

            override fun onEvent(eventType: Int, params: Bundle?) {}
        }

        speechRecognizer.setRecognitionListener(speechListener)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        } else {
            resetRecognizer()
        }
    }

    private fun stopCamera() = stopRealsensePipeline()

//    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
//        Log.d(TAG, "YOLO DETECTION: ${boundingBoxes.size} boxes detected for mode: $currentMode")
//
//        runOnUiThread {
//            binding.inferenceTime.text = "${inferenceTime}ms"
//
//            if (!isTTSInitialized) {
//                binding.overlay.setResults(boundingBoxes)
//                isDetectorBusy = false
//                return@runOnUiThread
//            }
//
//            val currentTime = System.currentTimeMillis()
//            val highConfidenceBoxes = boundingBoxes.filter { it.confidence > minConfidence }
//
//            // Calculate distances
//            val depthWidth = currentDepthFrame?.width ?: 0
//            val depthHeight = currentDepthFrame?.height ?: 0
//
//            Log.d(TAG, "Depth frame info: ${depthWidth}x${depthHeight}")
//
//            if (depthWidth > 0 && depthHeight > 0 && currentDepthFrame != null) {
//                highConfidenceBoxes.forEach { box ->
//                    val distance = calculateRobustDistance(box, depthWidth, depthHeight)
//                    box.distance = distance
//                    Log.d(TAG, "Object: ${box.clsName} -> ${distance}m")
//                }
//            } else {
//                Log.w(TAG, "No depth data available")
//                highConfidenceBoxes.forEach { box ->
//                    box.distance = -1f
//                }
//            }
//
//            // Update overlay
//            binding.overlay.setResults(highConfidenceBoxes)
//
//            // Handle different detection modes
//            when(currentMode) {
//                DetectionMode.START_DETECTION -> {
//                    if (isDetectionActive && depthWidth > 0 && depthHeight > 0) {
//                        handleStartDetection(highConfidenceBoxes, currentTime)
//                    }
//                }
//                DetectionMode.EXPLAIN_SURROUNDING -> {
//                    // Handle explain surrounding - only if we're in analysis mode
//                    if (isAnalyzingFrame) {
//                        Log.d(TAG, "Processing explain surrounding results - completing analysis")
//                        handleExplainSurroundingResults(highConfidenceBoxes, depthWidth > 0 && depthHeight > 0)
//
//                        // Reset analysis state after processing
//                        resetAnalysisState()
//                    } else {
//                        Log.d(TAG, "Received detection results but not in analysis mode - ignoring")
//                    }
//                }
//                DetectionMode.READ_MODE -> {
//                    // Read mode doesn't need object detection processing
//                }
//            }
//
//            // Reset detector busy flag
//            isDetectorBusy = false
//        }
//    }

    override fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long) {
        Log.d(TAG, "YOLO DETECTION: ${boundingBoxes.size} boxes detected for mode: $currentMode")

        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"

            if (!isTTSInitialized) {
                // Still show all boxes in overlay for debugging, but filter invalid distances
                val validBoxes = boundingBoxes.filter { it.distance > 0f }
                binding.overlay.setResults(validBoxes)
                isDetectorBusy = false
                return@runOnUiThread
            }

            val currentTime = System.currentTimeMillis()
            val highConfidenceBoxes = boundingBoxes.filter { it.confidence > minConfidence }

            // Calculate distances
            val depthWidth = currentDepthFrame?.width ?: 0
            val depthHeight = currentDepthFrame?.height ?: 0

            Log.d(TAG, "Depth frame info: ${depthWidth}x${depthHeight}")

            if (depthWidth > 0 && depthHeight > 0 && currentDepthFrame != null) {
                highConfidenceBoxes.forEach { box ->
                    val distance = calculateRobustDistance(box, depthWidth, depthHeight)
                    box.distance = distance
                    Log.d(TAG, "Object: ${box.clsName} -> ${distance}m")
                }
            } else {
                Log.w(TAG, "No depth data available")
                highConfidenceBoxes.forEach { box ->
                    box.distance = -1f
                }
            }

            // Filter out objects without valid distance data for display and processing
            val validBoxes = highConfidenceBoxes.filter { it.distance > 0f }
            val invalidBoxes = highConfidenceBoxes.filter { it.distance <= 0f }

            if (invalidBoxes.isNotEmpty()) {
                Log.d(TAG, "Filtered out ${invalidBoxes.size} objects without valid distance data")
            }

            // Update overlay with only valid boxes
            binding.overlay.setResults(validBoxes)

            // Handle different detection modes with only valid distance objects
            when(currentMode) {
                DetectionMode.START_DETECTION -> {
                    if (isDetectionActive && depthWidth > 0 && depthHeight > 0 && validBoxes.isNotEmpty()) {
                        handleStartDetection(validBoxes, currentTime)
                    }
                }
                DetectionMode.EXPLAIN_SURROUNDING -> {
                    if (isAnalyzingFrame) {
                        Log.d(TAG, "Processing explain surrounding results - completing analysis")
                        handleExplainSurroundingResults(validBoxes, depthWidth > 0 && depthHeight > 0)
                        resetAnalysisState()
                    } else {
                        Log.d(TAG, "Received detection results but not in analysis mode - ignoring")
                    }
                }
                DetectionMode.READ_MODE -> {
                    // Read mode doesn't need object detection processing
                }
            }

            // Reset detector busy flag
            isDetectorBusy = false
        }
    }

    private fun handleExplainSurroundingResults(detectedBoxes: List<BoundingBox>, hasDepthData: Boolean) {
        Log.d(TAG, "Processing explain surrounding results: ${detectedBoxes.size} objects with valid distances")

        if (detectedBoxes.isEmpty()) {
            if (hasDepthData) {
                speak("Environment scan complete. No objects with reliable distance data detected. Area appears clear.")
            } else {
                speak("Environment scan complete. No depth data available. Hold camera steady and try again.")
            }
            return
        }

        // All objects passed here have valid distances (> 0f), so proceed with detailed description
        val description = buildDetailedEnvironmentDescription(detectedBoxes)
        speak(description)
    }

//    private fun handleExplainSurroundingResults(detectedBoxes: List<BoundingBox>, hasDepthData: Boolean) {
//        Log.d(TAG, "Processing explain surrounding results: ${detectedBoxes.size} objects, depth: $hasDepthData")
//
//        if (detectedBoxes.isEmpty()) {
//            speak("Environment scan complete. No objects detected. Path appears clear in all directions.")
//            return
//        }
//
//        if (!hasDepthData) {
//            val objectNames = detectedBoxes.map { getSpecificObjectName(it.clsName) }.distinct()
//            speak("Objects visible: ${objectNames.joinToString(", ")}. Unable to determine distances. Please scan with steady camera movement.")
//            return
//        }
//
//        // Filter objects with valid distance data
//        val objectsWithDistance = detectedBoxes.filter { it.distance > 0f }
//
//        if (objectsWithDistance.isEmpty()) {
//            val objectNames = detectedBoxes.map { getSpecificObjectName(it.clsName) }.distinct()
//            speak("No objects detected. Recommend cautious movement.")
//            return
//        }
//
//        // Build enhanced description with detailed positioning and navigation guidance
//        val description = buildDetailedEnvironmentDescription(objectsWithDistance)
//        speak(description)
//    }

    private fun buildDetailedEnvironmentDescription(objects: List<BoundingBox>): String {
        val frameWidth = 640
        val frameHeight = 480

        // Get enhanced positions for all objects
        val objectsWithEnhancedPositions = objects.map { box ->
            val position = getEnhancedObjectPosition(box, frameWidth, frameHeight)
            Pair(box, position)
        }

        // Prioritize objects by safety relevance (convert to regular ObjectPosition for existing function)
        val objectsWithRegularPositions = objectsWithEnhancedPositions.map { (box, enhancedPos) ->
            val regularPos = ObjectPosition(
                distance = enhancedPos.distance,
                horizontalPosition = when {
                    enhancedPos.detailedHorizontalPosition.contains("left") -> "left"
                    enhancedPos.detailedHorizontalPosition.contains("right") -> "right"
                    else -> "center"
                },
                verticalPosition = when {
                    enhancedPos.detailedVerticalPosition.contains("upper") -> "upper"
                    enhancedPos.detailedVerticalPosition.contains("lower") -> "lower"
                    else -> "middle"
                },
                angle = enhancedPos.angle
            )
            Pair(box, regularPos)
        }
        val prioritizedObjects = prioritizeObjects(objectsWithRegularPositions)

        // Separate by threat level for navigation (using regular positions)
        val immediate = prioritizedObjects.filter { it.first.distance < 2.25f } // Within 3 steps
        val nearTerm = prioritizedObjects.filter { it.first.distance >= 2.25f && it.first.distance < 7.5f } // 3-10 steps
        val distant = prioritizedObjects.filter { it.first.distance >= 7.5f } // Beyond 10 steps

        val description = buildString {
            append("Environment scan: ")

            // Critical immediate obstacles (highest priority)
            if (immediate.isNotEmpty()) {
                append("Immediate attention required. ")
                immediate.take(3).forEachIndexed { index, (box, _) ->
                    // Get enhanced position for this specific box for detailed description
                    val enhancedPos = getEnhancedObjectPosition(box, frameWidth, frameHeight)
                    val objectName = getSpecificObjectName(box.clsName)
                    val distance = convertDistanceToSteps(box.distance)
                    val locationDesc = "${enhancedPos.detailedHorizontalPosition} ${enhancedPos.detailedVerticalPosition}".trim()

                    append("$objectName $distance at $locationDesc")
                    if (box.distance < 0.75f) append(" - very close")
                    if (index < immediate.size - 1 && index < 2) append(", ")
                }
                append(". ")
            }

            // Near-term navigation objects
            if (nearTerm.isNotEmpty()) {
                val nearObjects = nearTerm.take(4)
                append("Nearby obstacles: ")
                nearObjects.forEachIndexed { index, (box, _) ->
                    // Get enhanced position for detailed description
                    val enhancedPos = getEnhancedObjectPosition(box, frameWidth, frameHeight)
                    val objectName = getSpecificObjectName(box.clsName)
                    val distance = convertDistanceToSteps(box.distance)
                    val direction = enhancedPos.angleDescription

                    append("$objectName $distance $direction")
                    if (index < nearObjects.size - 1) append(", ")
                }
                append(". ")
            }

            // Distant landmarks (for orientation)
            if (distant.isNotEmpty()) {
                val landmarks = distant.take(3).map { getSpecificObjectName(it.first.clsName) }.distinct()
                append("Distant landmarks: ${landmarks.joinToString(", ")}. ")
            }

            // Generate and include navigation recommendations
            val safeRoutes = generateSafeRoutes(objectsWithRegularPositions)
            if (safeRoutes.isNotEmpty()) {
                val primaryRoute = safeRoutes.first()
                append("Navigation guidance: ${primaryRoute.description}")

                // Add alternative if available and different
                if (safeRoutes.size > 1 && safeRoutes[1].direction != primaryRoute.direction) {
                    append(" Alternative: ${safeRoutes[1].description}")
                }
            } else {
                // Provide general guidance when no specific routes generated
                when {
                    immediate.isEmpty() && nearTerm.isEmpty() ->
                        append("All directions appear clear for safe movement.")
                    immediate.any { it.second.horizontalPosition == "center" } ->
                        append("Center path blocked. Side movement recommended.")
                    else ->
                        append("Exercise caution and move slowly.")
                }
            }
        }

        return description
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.invalidate()

            // Handle empty detection for single frame analysis
            if (currentMode == DetectionMode.EXPLAIN_SURROUNDING && isAnalyzingFrame) {
                Log.d(TAG, "Empty detection for explain surrounding")
                speak("No objects detected in current view. Area appears clear.")
                resetAnalysisState()
            }

            // Reset detector busy flag
            isDetectorBusy = false
        }
    }

    private fun calculateRobustDistance(box: BoundingBox, depthWidth: Int, depthHeight: Int): Float {
        val depthFrame = currentDepthFrame ?: return -1f  // Use -1f to indicate no depth data

        Log.d(TAG, "=== Distance Calculation Debug ===")
        Log.d(TAG, "Object: ${box.clsName}, Confidence: ${"%.3f".format(box.confidence)}")
        Log.d(TAG, "Depth frame dimensions: ${depthWidth}x${depthHeight}")
        Log.d(TAG, "Normalized box coordinates: (${box.x1}, ${box.y1}) to (${box.x2}, ${box.y2})")

        // Check for valid depth frame dimensions
        if (depthWidth <= 0 || depthHeight <= 0) {
            Log.e(TAG, "Invalid depth frame dimensions: ${depthWidth}x${depthHeight}")
            return -1f
        }

        // Convert normalized coordinates to depth frame coordinates with bounds checking
        val leftX = (box.x1 * depthWidth).toInt().coerceIn(0, depthWidth - 1)
        val rightX = (box.x2 * depthWidth).toInt().coerceIn(0, depthWidth - 1)
        val topY = (box.y1 * depthHeight).toInt().coerceIn(0, depthHeight - 1)
        val bottomY = (box.y2 * depthHeight).toInt().coerceIn(0, depthHeight - 1)

        Log.d(TAG, "Depth frame coordinates: ($leftX, $topY) to ($rightX, $bottomY)")

        // Ensure we have a valid bounding box
        if (rightX <= leftX || bottomY <= topY) {
            Log.e(TAG, "Invalid bounding box after coordinate conversion")
            return -1f
        }

        // Sample strategic points within the bounding box
        val samplePoints = mutableListOf<Pair<Int, Int>>()

        // Center point
        val centerX = (leftX + rightX) / 2
        val centerY = (topY + bottomY) / 2
        samplePoints.add(Pair(centerX, centerY))

        // Bottom center (often most reliable for ground-based objects)
        val bottomCenterX = centerX
        val bottomCenterY = bottomY - ((bottomY - topY) * 0.2f).toInt() // 20% up from bottom
        samplePoints.add(Pair(bottomCenterX, bottomCenterY))

        // Additional strategic points for larger objects
        val boxWidth = rightX - leftX
        val boxHeight = bottomY - topY

        if (boxWidth > 20 && boxHeight > 20) { // Only for reasonably sized boxes
            // Left and right center points
            samplePoints.add(Pair(leftX + boxWidth / 4, centerY))
            samplePoints.add(Pair(rightX - boxWidth / 4, centerY))

            // Top center (for tall objects)
            samplePoints.add(Pair(centerX, topY + boxHeight / 4))
        }

        // Collect valid distance measurements
        val validDistances = mutableListOf<Float>()

        Log.d(TAG, "Sampling ${samplePoints.size} points for distance calculation:")

        for ((index, pair) in samplePoints.withIndex()) {
            val (x, y) = pair
            try {
                // Additional bounds check before accessing depth data
                if (x < 0 || x >= depthWidth || y < 0 || y >= depthHeight) {
                    Log.w(TAG, "  Point $index at ($x,$y): Out of bounds, skipping")
                    continue
                }

                val distance = depthFrame.getDistance(x, y)

                Log.d(TAG, "  Point $index at ($x,$y): ${distance}m")

                // More strict filtering for realistic distances
                when {
                    distance <= 0.05f -> {
                        Log.d(TAG, "    -> Rejected: Too close (likely sensor noise)")
                    }
                    distance > 15.0f -> {
                        Log.d(TAG, "    -> Rejected: Too far (likely invalid reading)")
                    }
                    else -> {
                        validDistances.add(distance)
                        Log.d(TAG, "    -> Accepted: ${distance}m")
                    }
                }
            } catch (e: Exception) {
                Log.w(TAG, "  Point $index at ($x,$y): Failed to get distance", e)
            }
        }

        val finalDistance = when {
            validDistances.isEmpty() -> {
                Log.w(TAG, "No valid distance readings for ${box.clsName}")
                -1f  // Indicate no valid data
            }
            validDistances.size == 1 -> {
                Log.d(TAG, "Single distance reading: ${validDistances.first()}m")
                validDistances.first()
            }
            else -> {
                // Use median for stability, but also check for outliers
                val sorted = validDistances.sorted()
                val median = if (sorted.size % 2 == 0) {
                    (sorted[sorted.size / 2 - 1] + sorted[sorted.size / 2]) / 2f
                } else {
                    sorted[sorted.size / 2]
                }

                // Check if readings are consistent (within 30% of median)
                val consistentReadings = sorted.filter {
                    val percentDiff = kotlin.math.abs(it - median) / median
                    percentDiff < 0.3f
                }

                val result = if (consistentReadings.size >= validDistances.size * 0.6f) {
                    // Most readings are consistent, use median
                    median
                } else {
                    // Readings are inconsistent, use the most common range
                    Log.w(TAG, "Inconsistent distance readings, using closest reliable reading")
                    sorted.first() // Use closest reading as most reliable
                }

                Log.d(TAG, "Multiple readings: ${validDistances.map { "%.2f".format(it) }}")
                Log.d(TAG, "Final result: ${result}m (median: ${median}m)")
                result
            }
        }

        Log.d(TAG, "=== Final distance for ${box.clsName}: ${finalDistance}m ===")
        return finalDistance
    }

    private fun safeVibrate() {
        try {
            vibrator.vibrate(VibrationEffect.createOneShot(50, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (e: Exception) {
            Log.e(TAG, "Vibration error", e)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
                resetRecognizer()
            } else {
                Log.e(TAG, "Permissions not granted")
            }
        }
    }

    override fun onDestroy() {
        unregisterReceiver(usbReceiver)
        rsContext.close()
        super.onDestroy()

        detector.clear()

        audioExecutor.shutdown()
        speechExecutor.shutdown()
        cameraExecutor.shutdown()
        detectionExecutor.shutdown()

        try {
            if (!audioExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                audioExecutor.shutdownNow()
            }
            if (!speechExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                speechExecutor.shutdownNow()
            }
            if (!cameraExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                cameraExecutor.shutdownNow()
            }
            if (!detectionExecutor.awaitTermination(2, TimeUnit.SECONDS)) {
                detectionExecutor.shutdownNow()
            }
        } catch (e: InterruptedException) {
            Thread.currentThread().interrupt()
        }

        textToSpeech.stop()
        textToSpeech.shutdown()
        speechRecognizer.destroy()
        audioManager.abandonAudioFocusRequest(audioFocusRequest)
        textRecognizer.close()
    }

    override fun onPause() {
        stopRealsensePipeline()
        stopListening()
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
        if (isDetectionActive && rsContext.queryDevices().deviceCount > 0 && !isPipelineRunning) {
            startRealsensePipeline()
        }
        safeStartListening()
    }

    companion object {
        private const val TAG = "ObjectDetection"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private const val REQUEST_CODE_AUDIO_PERMISSION = 123
        private val REQUIRED_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        )
    }
}