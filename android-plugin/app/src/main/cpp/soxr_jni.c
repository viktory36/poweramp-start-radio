/*
 * JNI wrapper for libsoxr resampling.
 * Provides a single function: resample(float[], int, int) -> float[]
 */
#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <math.h>
#include "soxr.h"

#define TAG "SoxrJNI"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

JNIEXPORT jfloatArray JNICALL
Java_com_powerampstartradio_indexing_NativeResampler_nativeResample(
    JNIEnv *env,
    jclass clazz,
    jfloatArray inputArray,
    jint fromRate,
    jint toRate,
    jint quality)
{
    jsize inputLen = (*env)->GetArrayLength(env, inputArray);
    if (inputLen == 0 || fromRate == toRate) {
        return inputArray;
    }

    /* Get input samples */
    jfloat *input = (*env)->GetFloatArrayElements(env, inputArray, NULL);
    if (!input) {
        LOGE("Failed to get input array elements");
        return NULL;
    }

    /* Compute output length */
    double ratio = (double)toRate / (double)fromRate;
    jsize outputLen = (jsize)ceil((double)inputLen * ratio);

    /* Allocate output buffer */
    float *output = (float *)malloc(outputLen * sizeof(float));
    if (!output) {
        (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);
        LOGE("Failed to allocate output buffer (%d samples)", outputLen);
        return NULL;
    }

    /* Map quality parameter to soxr recipe */
    unsigned long recipe;
    switch (quality) {
        case 0: recipe = SOXR_MQ; break;
        case 1: recipe = SOXR_HQ; break;
        case 2: recipe = SOXR_VHQ; break;
        default: recipe = SOXR_HQ; break;
    }

    /* Configure soxr */
    soxr_quality_spec_t q_spec = soxr_quality_spec(recipe, 0);
    soxr_io_spec_t io_spec = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);

    /* One-shot resample */
    size_t idone, odone;
    soxr_error_t error = soxr_oneshot(
        (double)fromRate, (double)toRate, 1,  /* mono */
        input, (size_t)inputLen, &idone,
        output, (size_t)outputLen, &odone,
        &io_spec, &q_spec, NULL);

    (*env)->ReleaseFloatArrayElements(env, inputArray, input, JNI_ABORT);

    if (error) {
        LOGE("soxr_oneshot failed: %s", soxr_strerror(error));
        free(output);
        return NULL;
    }

    /* Create Java array with actual output length */
    jfloatArray result = (*env)->NewFloatArray(env, (jsize)odone);
    if (!result) {
        LOGE("Failed to create output Java array");
        free(output);
        return NULL;
    }
    (*env)->SetFloatArrayRegion(env, result, 0, (jsize)odone, output);
    free(output);

    return result;
}
