package com.example.android.camerax.tflite

import android.graphics.Point
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import kotlin.math.pow
import kotlin.math.sqrt

fun Mat.toMatOfPoint(dst: MatOfPoint) {
    convertTo(dst, CvType.CV_32SC2)
}

fun Mat.toMatOfPoint2F(dst: MatOfPoint2f) {
    convertTo(dst, CvType.CV_32FC2)
}

fun <T : Mat> MutableCollection<T>.release() {
    iterator().let { iterator ->
        while (iterator.hasNext()) {
            iterator.next().release()
            iterator.remove()
        }
    }
}

fun hypotenuse(
    a: Double,
    b: Double,
): Double = sqrt(a.pow(2) + b.pow(2))

fun hypotenuse(
    a: Float,
    b: Float,
): Float = sqrt(a.pow(2) + b.pow(2))

fun hypotenuse(
    a: Float,
    b: Float,
    c: Float,
): Float = sqrt(a.pow(2) + b.pow(2) + c.pow(2))

fun MatOfPoint.toFloatArray(): FloatArray {
    val result = FloatArray(rows() * 2)
    val intArray = intArrayOf(0, 0)
    var index = 0

    (0 until rows()).map { x ->
        get(x, 0, intArray)
        result[index] = intArray[0].toFloat()
        result[index + 1] = intArray[1].toFloat()
        index += 2
    }

    return result
}

fun MatOfPoint.toPoints(): List<Point> {
    val result = mutableListOf<Point>()
    val intArray = intArrayOf(0, 0)

    (0 until rows()).map { x ->
        get(x, 0, intArray)
        result.add(Point(intArray[0], intArray[1]))
    }

    return result.toList()
}

fun ByteArray.toMat(
    width: Int,
    height: Int,
): Mat {
    val result = Mat(height, width, CvType.CV_8UC1)
    result.put(0, 0, this)
    return result
}

fun Mat.print(tag: String) {
    val byteArray = byteArrayOf(0)

    (0 until rows()).forEach { y ->
        val string = StringBuilder("$y: ")
        (0 until cols()).forEach { x ->
            get(x, y, byteArray)
            string.append(byteArray[0].toInt()).append(" ")
        }
        Log.d(tag, string.toString())
    }
}

fun MatOfPoint2f.print(tag: String) {
    val floatArray = floatArrayOf(0f, 0f)
    val string = StringBuilder()

    (0 until cols()).forEach { x ->
        (0 until rows()).forEach { y ->
            get(x, y, floatArray)
            string.append("(${floatArray[0]}, ${floatArray[1]}) ")
        }
    }
    Log.d(tag, string.toString())
}

// Sorts the points by tl, tr, br, bl
// https://github.com/PyImageSearch/imutils/blob/master/imutils/perspective.py
fun Mat.orderPoints() {
    val x =
        listOf(
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
            floatArrayOf(0f, 0f),
        )

    get(0, 0, x[0])
    get(1, 0, x[1])
    get(2, 0, x[2])
    get(3, 0, x[3])

    val xSorted = x.sortedBy { it[0] }

    val leftMost = xSorted.subList(0, 2)
    val rightMost = xSorted.subList(2, 4)

    val leftMostSorted =
        leftMost
            .sortedBy { it[1] }

    val tl = leftMostSorted[0]
    val bl = leftMostSorted[1]

    val rightMostSorted =
        rightMost
            .sortedBy {
                hypotenuse(
                    it[0] - tl[0],
                    it[1] - tl[1],
                )
            }

    val tr = rightMostSorted[0]
    val br = rightMostSorted[1]

    reshape(1, 4)
    put(0, 0, tl)
    put(1, 0, tr)
    put(2, 0, br)
    put(3, 0, bl)
}
