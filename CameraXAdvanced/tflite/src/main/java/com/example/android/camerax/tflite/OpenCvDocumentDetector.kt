package com.example.android.camerax.tflite

import android.graphics.Point
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Find the corners of the largest rectangular object using OpenCV Contour detection.
 *
 * Based on:
 * https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
 */
class OpenCvDocumentDetector {
    // Temporary data structures to be re-used for performance purposes
    private lateinit var hierarchy: Mat

    private lateinit var contour: MatOfPoint
    private lateinit var kernel: Mat

    private val contours = mutableListOf<MatOfPoint>()

    private var currentContour: MatOfPoint? = null
    private val blurSize = Size(5.0, 5.0)

    private var isProcessing = false

    fun initialize() {
        hierarchy = Mat()
        contour = MatOfPoint()
        kernel =
            Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                blurSize,
            )
    }

    fun release() {
        if (::hierarchy.isInitialized) hierarchy.release()
        if (::kernel.isInitialized) kernel.release()

        if (::contour.isInitialized) contour.release()
        currentContour = null

        contours.release()
    }

    fun detect(
        foregroundMask: ByteArray,
        width: Int,
        height: Int,
    ): List<Point> {
        if (isProcessing) {
            Log.d("carlos", "dropped frame")
            return currentContour?.toPoints() ?: emptyList()
        }

        isProcessing = true

        Log.d("carlos", "foregroundMask.toMat")
        val maskedFrame = foregroundMask.toMat(width, height)

        // Find edge contours
        Log.d("carlos", "findedges")
        Imgproc.findContours(
            maskedFrame,
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE,
        )

        val frameArea = maskedFrame.rows() * maskedFrame.cols()

        // Find the largest contour with 4 corners
        Log.d("carlos", "findcorners")
        val contourFloats =
            contours.map { contour ->
                val contourFloat = MatOfPoint2f()
                contour.toMatOfPoint2F(contourFloat)

                // Approximate the contour
                val peri =
                    Imgproc.arcLength(
                        contourFloat,
                        true,
                    )

                Imgproc.approxPolyDP(
                    contourFloat,
                    contourFloat,
                    0.02 * peri,
                    true,
                )

                contourFloat
            }

        contourFloats
            .filter { it.rows() == 4 }
            .maxByOrNull { Imgproc.contourArea(it) }
            ?.let { screenContour ->
                Log.d("carlos", "found screenContour ${screenContour.rows()} points, ${Imgproc.contourArea(screenContour)} / $frameArea area")
                screenContour.orderPoints()
                screenContour.toMatOfPoint(contour)

                currentContour = contour
            }
            ?: run {
                currentContour = null
            }

        contours.release()
        contourFloats.forEach { it.release() }
        maskedFrame.release()
        isProcessing = false

        Log.d("carlos", "done")
        return currentContour?.toPoints() ?: emptyList()
    }
}
