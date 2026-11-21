/*
 * q2_serial.c - Serial Edge Detection Using Sobel Operator
 *
 * Applies 3×3 Sobel filters (Gx, Gy) to an n×n synthetic grayscale image.
 * Gradient magnitude: G = sqrt(Gx² + Gy²), clamped to [0, 255].
 *
 * Compile: gcc -o q2_serial q2_serial.c -lm
 * Run:     ./q2_serial [image_size]
 *          Default image_size = 4096
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
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

/* Sobel 3×3 kernels */
static const int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

static const int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

int main(int argc, char *argv[]) {
    int n = 4096; /* Default image size */
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n < 3) n = 3;
    }

    printf("Serial Edge Detection (Sobel) on %d x %d image\n", n, n);

    /* Allocate image and output */
    unsigned char *image = (unsigned char *)malloc(n * n * sizeof(unsigned char));
    unsigned char *output = (unsigned char *)calloc(n * n, sizeof(unsigned char));
    if (!image || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Initialize synthetic image with a pattern (diagonal gradient + noise) */
    unsigned int seed = 123;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            /* Diagonal gradient with some variation */
            image[i * n + j] = (unsigned char)(((i + j) * 255) / (2 * n - 2));
            /* Add some structure: horizontal and vertical lines */
            if (i % 64 == 0 || j % 64 == 0) {
                image[i * n + j] = 255;
            }
        }
    }

    printf("Image initialized.\n");

    /* Apply Sobel edge detection */
    double start = get_time();

    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            int sum_x = 0, sum_y = 0;

            /* Convolve with Sobel kernels */
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = image[(i + ki) * n + (j + kj)];
                    sum_x += Gx[ki + 1][kj + 1] * pixel;
                    sum_y += Gy[ki + 1][kj + 1] * pixel;
                }
            }

            /* Gradient magnitude */
            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            if (magnitude > 255) magnitude = 255;
            output[i * n + j] = (unsigned char)magnitude;
        }
    }

    double end = get_time();
    double elapsed = end - start;

    printf("Edge detection complete.\n");
    printf("Execution time: %.6f seconds\n", elapsed);

    /* Print a small sample of the output for verification */
    printf("\nSample output pixels (row, col, value):\n");
    int step = n / 8;
    if (step < 1) step = 1;
    for (int i = 1; i < n - 1 && i < 10 * step; i += step) {
        for (int j = 1; j < n - 1 && j < 10 * step; j += step) {
            printf("  (%4d, %4d) = %3d\n", i, j, output[i * n + j]);
        }
    }

    free(image);
    free(output);

    return 0;
}
