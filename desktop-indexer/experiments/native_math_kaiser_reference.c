#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Experiment-only scalar extraction of android-plugin/app/src/main/cpp/math_jni.c.
 * Keep coefficient generation, phase layout, boundary padding, and output length
 * in sync with NativeMath. This intentionally excludes JNI and NEON reduction order.
 */

typedef struct {
    int up;
    int down;
    int taps;
    int center_tap;
    float *phases;
} polyphase_plan;

static double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double y = x * x * 0.25;
    for (int k = 1; k <= 30; k++) {
        term *= y / ((double)k * k);
        sum += term;
        if (term < sum * 1e-16) break;
    }
    return sum;
}

static int gcd_int(int a, int b) {
    while (b) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static void free_plan(polyphase_plan *plan) {
    free(plan->phases);
    plan->phases = NULL;
}

static int build_plan(int from_rate, int to_rate, polyphase_plan *plan) {
    memset(plan, 0, sizeof(*plan));
    if (from_rate <= 0 || to_rate <= 0) return 0;

    int g = gcd_int(from_rate, to_rate);
    plan->up = to_rate / g;
    plan->down = from_rate / g;
    int max_rate = plan->up > plan->down ? plan->up : plan->down;
    if (max_rate > (INT_MAX - 1) / 20) return 0;

    double cutoff = 1.0 / max_rate;
    int half_len = 10 * max_rate;
    int filter_length = 2 * half_len + 1;
    float *filter = malloc((size_t)filter_length * sizeof(float));
    if (!filter) return 0;

    const double beta = 5.0;
    double inverse_i0 = 1.0 / bessel_i0(beta);
    double half = (double)half_len;
    for (int n = 0; n < filter_length; n++) {
        double t = n - half;
        double sinc = fabs(t) < 1e-10
            ? cutoff
            : sin(M_PI * cutoff * t) / (M_PI * t);
        double r = t / half;
        double window_argument = 1.0 - r * r;
        double window = window_argument > 0.0
            ? bessel_i0(beta * sqrt(window_argument)) * inverse_i0
            : 0.0;
        filter[n] = (float)(sinc * window * plan->up);
    }

    plan->taps = filter_length / plan->up + (filter_length % plan->up != 0);
    size_t phase_count = (size_t)plan->up * (size_t)plan->taps;
    if (plan->up > 0 && phase_count / (size_t)plan->up != (size_t)plan->taps) {
        free(filter);
        return 0;
    }
    plan->phases = calloc(phase_count, sizeof(float));
    if (!plan->phases) {
        free(filter);
        return 0;
    }
    for (int phase = 0; phase < plan->up; phase++) {
        for (int tap = 0; tap < plan->taps; tap++) {
            int filter_index = phase + tap * plan->up;
            if (filter_index < filter_length) {
                plan->phases[phase * plan->taps + tap] = filter[filter_index];
            }
        }
    }
    free(filter);

    for (int phase = 0; phase < plan->up; phase++) {
        float *phase_taps = &plan->phases[phase * plan->taps];
        for (int tap = 0; tap < plan->taps / 2; tap++) {
            int opposite = plan->taps - 1 - tap;
            float temporary = phase_taps[tap];
            phase_taps[tap] = phase_taps[opposite];
            phase_taps[opposite] = temporary;
        }
    }
    plan->center_tap = (plan->taps - 1) / 2;
    return 1;
}

int64_t native_math_output_length(int64_t input_length, int from_rate, int to_rate) {
    if (input_length < 0 || from_rate <= 0 || to_rate <= 0) return -1;
    int g = gcd_int(from_rate, to_rate);
    int64_t up = to_rate / g;
    int64_t down = from_rate / g;
    if (input_length > (INT64_MAX - (down - 1)) / up) return -1;
    return (input_length * up + down - 1) / down;
}

int native_math_kaiser_resample(
    const float *input,
    int64_t input_length,
    int from_rate,
    int to_rate,
    float *output,
    int64_t output_length)
{
    if (!input || !output || input_length <= 0 || input_length > INT_MAX) return 0;
    int64_t expected = native_math_output_length(input_length, from_rate, to_rate);
    if (expected <= 0 || output_length != expected || output_length > INT_MAX) return 0;
    if (from_rate == to_rate) {
        memcpy(output, input, (size_t)input_length * sizeof(float));
        return 1;
    }

    polyphase_plan plan;
    if (!build_plan(from_rate, to_rate, &plan)) return 0;
    for (int64_t n = 0; n < output_length; n++) {
        int64_t position = n * plan.down;
        int phase = (int)(position % plan.up);
        int64_t source_start = position / plan.up - plan.center_tap;
        const float *phase_taps = &plan.phases[phase * plan.taps];
        float sum = 0.0f;
        for (int tap = 0; tap < plan.taps; tap++) {
            int64_t source_index = source_start + tap;
            if (source_index >= 0 && source_index < input_length) {
                sum += phase_taps[tap] * input[source_index];
            }
        }
        output[n] = sum;
    }
    free_plan(&plan);
    return 1;
}
