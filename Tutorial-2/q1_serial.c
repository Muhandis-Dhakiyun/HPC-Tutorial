/*
 * q1_serial.c - Serial Monte Carlo Estimation of π
 *
 * Method: Generate N random (x,y) points in [0,1)×[0,1).
 *         Count how many fall inside the quarter-circle x²+y² ≤ 1.
 *         π ≈ 4 * (count / N).
 *
 * Compile: gcc -o q1_serial q1_serial.c -lm
 * Run:     ./q1_serial
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
static inline int rand_r(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((unsigned int)(*seed / 65536) % 32768);
}
#endif

#ifdef _WIN32
#include <windows.h>
/* High-resolution timer for Windows */
static double get_time(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}
#endif

int main(void) {
    /* Sample sizes to test */
    long sample_sizes[] = {10, 100, 10000, 1000000, 10000000, 100000000, 1000000000L};
    int num_sizes = sizeof(sample_sizes) / sizeof(sample_sizes[0]);

    printf("=============================================================\n");
    printf("     Serial Monte Carlo Estimation of Pi\n");
    printf("=============================================================\n");
    printf("%-14s %-14s %-14s %-14s\n",
           "N", "Pi Estimate", "Error", "Time (s)");
    printf("-------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        long N = sample_sizes[s];
        long count = 0;
        unsigned int seed = 42; /* Fixed seed for reproducibility */

        double start = get_time();

        for (long i = 0; i < N; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) {
                count++;
            }
        }

        double end = get_time();
        double elapsed = end - start;

        double pi_estimate = 4.0 * (double)count / (double)N;
        double error = fabs(pi_estimate - M_PI);

        printf("%-14ld %-14.10f %-14.10f %-14.6f\n",
               N, pi_estimate, error, elapsed);
    }

    printf("=============================================================\n");
    printf("Reference Pi = %.15f\n", M_PI);

    return 0;
}
