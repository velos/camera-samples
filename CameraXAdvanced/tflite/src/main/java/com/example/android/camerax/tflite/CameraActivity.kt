/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camerax.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup.MarginLayoutParams
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.updateLayoutParams
import androidx.lifecycle.LifecycleOwner
import com.android.example.camerax.tflite.databinding.ActivityCameraBinding
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter.RegionOfInterest
import org.opencv.android.OpenCVLoader
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.jvm.optionals.getOrNull
import kotlin.math.min
import kotlin.random.Random


/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    private lateinit var activityCameraBinding: ActivityCameraBinding

    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0

    private val imageSegmenter: InteractiveSegmenter by lazy {
        val imageSegmenterOptions =
            InteractiveSegmenter.InteractiveSegmenterOptions
                .builder()
                .setBaseOptions(
                    BaseOptions
                        .builder()
                        .setDelegate(Delegate.CPU)
                        .setModelAssetPath("magic_touch.tflite")
                        .build(),
                ).setOutputCategoryMask(true)
                .setOutputConfidenceMasks(false)
                .build()

        InteractiveSegmenter.createFromOptions(this, imageSegmenterOptions)
    }
    private val openCvDocumentDetector: OpenCvDocumentDetector = OpenCvDocumentDetector()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (savedInstanceState == null) {
            OpenCVLoader.initLocal()
        }

        openCvDocumentDetector.initialize()

        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)

        activityCameraBinding.cameraCaptureButton.setOnClickListener {

            // Disable all camera controls
            it.isEnabled = false

            if (pauseAnalysis) {
                // If image analysis is in paused state, resume it
                pauseAnalysis = false
                activityCameraBinding.imagePredicted.visibility = View.GONE

            } else {
                // Otherwise, pause image analysis and freeze image
                pauseAnalysis = true
                val matrix = Matrix().apply {
                    postRotate(imageRotationDegrees.toFloat())
                    if (isFrontFacing) postScale(-1f, 1f)
                }
                val uprightImage = Bitmap.createBitmap(
                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
                activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
                activityCameraBinding.imagePredicted.visibility = View.VISIBLE
            }

            // Re-enable camera controls
            it.isEnabled = true
        }
    }

    override fun onDestroy() {

        // Terminate all outstanding analyzing jobs (if there is any).
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }

        openCvDocumentDetector.release()

        // Release TFLite resources.
//        imageSegmenter.close()

        super.onDestroy()
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener ({

            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                // Early exit: image analysis is in paused state
                if (pauseAnalysis) {
                    image.close()
                    return@Analyzer
                }

                // Copy out RGB bits to our shared buffer
                image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)  }

                // Process the image in Tensorflow
                val predictionLocation = segmentImage(bitmapBuffer)//.rotate(imageRotationDegrees))

                // Report only the top prediction
                reportPrediction(predictionLocation, image.imageInfo.sensorToBufferTransformMatrix)

                // Compute the FPS of the entire pipeline
                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    val fps = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG, "FPS: ${"%.02f".format(fps)} with tensorSize: ${bitmapBuffer.width} x ${bitmapBuffer.height}")
                    lastFpsTimestamp = now
                }
            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this))
    }

    fun Bitmap.rotate(rotation: Int): Bitmap {
        if (rotation == 0) return this

        val matrix = Matrix()
        matrix.postRotate(rotation.toFloat())
        return Bitmap.createBitmap(this, 0, 0, width, height, matrix, false)
    }

    private fun segmentImage(bitmap: Bitmap): RectF? {
        val roi =
            RegionOfInterest.create(
                NormalizedKeypoint.create(
                    0.5f * bitmap.width,
                    0.5f * bitmap.height,
                ),
            )

        val mpImage = BitmapImageBuilder(bitmap).build()

        return imageSegmenter
            .segment(mpImage, roi)
            ?.categoryMask()
            ?.getOrNull()
            ?.let { segmentationResult ->
                val byteBuffer = ByteBufferExtractor.extract(segmentationResult)

                val foregroundMask = ByteArray(byteBuffer.capacity())
                byteBuffer.get(foregroundMask)
                for (i in foregroundMask.indices) {
                    foregroundMask[i] =
                        if (foregroundMask[i] < 0) {
                            0
                        } else {
                            -1 // Byte(-1) = Int(255)
                        }
                }

                val contourPoints =
                    openCvDocumentDetector.detect(
                        foregroundMask,
                        bitmap.width,
                        bitmap.height,
                    )

                if (contourPoints.size == 4) {
                    RectF(
                        contourPoints[3].x.toFloat(),
                        contourPoints[3].y.toFloat(),
                        contourPoints[1].x.toFloat(),
                        contourPoints[1].y.toFloat(),
                    )
                        .also {
                            Log.d("carlos", "contourPoints: $contourPoints ${bitmap.width} x ${bitmap.height}")
                        }
                } else {
                    null
                }
            }
    }

    private fun reportPrediction(
        predictionLocation: RectF?,
        sensorToBufferTransformMatrix: Matrix,
    ) = activityCameraBinding.viewFinder.post {

        // Early exit: if prediction is not good enough, don't report it
        if (predictionLocation == null) {
            activityCameraBinding.boxPrediction.visibility = View.GONE
            return@post
        }

        // Location has to be mapped to our local coordinates
        val location = mapOutputCoordinates(predictionLocation, sensorToBufferTransformMatrix)

        // Update the text and UI
        activityCameraBinding.boxPrediction.updateLayoutParams<MarginLayoutParams> {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(activityCameraBinding.viewFinder.width, location.right.toInt() - location.left.toInt())
            height = min(activityCameraBinding.viewFinder.height, location.bottom.toInt() - location.top.toInt())
        }

        // Make sure all UI elements are visible
        activityCameraBinding.boxPrediction.visibility = View.VISIBLE
    }

    /**
     * Helper function used to map the coordinates for objects coming out of
     * the model into the coordinates that the user sees on the screen.
     */
    private fun mapOutputCoordinates(
        location: RectF,
        sensorToAnalysis: Matrix,
    ): RectF {
        val correctedLocation = RectF()

        // Location has to be mapped to our local coordinates
        val sensorToView = activityCameraBinding.viewFinder.sensorToViewTransform ?: Matrix()
        val analysisToView = Matrix()
        sensorToAnalysis.invert(analysisToView)
        analysisToView.postConcat(sensorToView)
        analysisToView.mapRect(correctedLocation, location)

        Log.d("carlos", "analysisToView: $location => $correctedLocation ${activityCameraBinding.viewFinder.width} x ${activityCameraBinding.viewFinder.height}")

        return correctedLocation
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName
    }
}
