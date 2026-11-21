/*
 * q2_parallel.c - Parallel Edge Detection Using Pthreads (Sobel Operator)
 *
 * Divides image rows among threads. Each thread applies 3×3 Sobel filters
 * to its assigned rows. Rows are independent (each output pixel depends only
 * on its 3×3 neighborhood in the input image, which is read-only).
 *
 * Compile: gcc -o q2_parallel q2_parallel.c -lpthread -lm
 * Run:     ./q2_parallel [image_size]
 *          Default image_size = 4096
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

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

/* Shared data for all threads */
typedef struct {
    unsigned char *image;
    unsigned char *output;
    int n;          /* Image dimension */
    int start_row;  /* First row this thread processes */
    int end_row;    /* Last row (exclusive) this thread processes */
} thread_data_t;

/* Thread function: apply Sobel to assigned rows */
void *sobel_worker(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    int n = data->n;
    unsigned char *image = data->image;
    unsigned char *output = data->output;

    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 1; j < n - 1; j++) {
            int sum_x = 0, sum_y = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = image[(i + ki) * n + (j + kj)];
                    sum_x += Gx[ki + 1][kj + 1] * pixel;
                    sum_y += Gy[ki + 1][kj + 1] * pixel;
                }
            }

            int magnitude = (int)sqrt((double)(sum_x * sum_x + sum_y * sum_y));
            if (magnitude > 255) magnitude = 255;
            output[i * n + j] = (unsigned char)magnitude;
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    int n = 4096;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n < 3) n = 3;
    }

    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("===================================================================\n");
    printf("  Parallel Edge Detection (Sobel) on %d x %d image (Pthreads)\n", n, n);
    printf("===================================================================\n");

    /* Allocate image and output */
    unsigned char *image = (unsigned char *)malloc(n * n * sizeof(unsigned char));
    unsigned char *output = (unsigned char *)calloc(n * n, sizeof(unsigned char));
    if (!image || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Initialize synthetic image */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            image[i * n + j] = (unsigned char)(((i + j) * 255) / (2 * n - 2));
            if (i % 64 == 0 || j % 64 == 0) {
                image[i * n + j] = 255;
            }
        }
    }

    printf("Image initialized.\n\n");
    printf("%-10s %-14s %-12s %-12s\n", "Threads", "Time (s)", "Speedup", "Efficiency");
    printf("-------------------------------------------------------------------\n");

    double serial_time = -1.0;

    for (int t = 0; t < num_thread_counts; t++) {
        int num_threads = thread_counts[t];

        /* Clear output */
        memset(output, 0, n * n * sizeof(unsigned char));

        pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
        thread_data_t *tdata = malloc(num_threads * sizeof(thread_data_t));

        /* Rows to process: 1 to n-2 (skip boundary rows) */
        int total_rows = n - 2;
        int rows_per_thread = total_rows / num_threads;
        int extra_rows = total_rows % num_threads;

        int current_row = 1; /* Start from row 1 (skip row 0) */
        for (int i = 0; i < num_threads; i++) {
            tdata[i].image = image;
            tdata[i].output = output;
            tdata[i].n = n;
            tdata[i].start_row = current_row;
            int rows = rows_per_thread + (i < extra_rows ? 1 : 0);
            tdata[i].end_row = current_row + rows;
            current_row += rows;
        }

        double start = get_time();

        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, sobel_worker, &tdata[i]);
        }

        for (int i = 0; i < num_threads; i++) {
            pthread_join(threads[i], NULL);
        }

        double end = get_time();
        double elapsed = end - start;

        if (num_threads == 1) {
            serial_time = elapsed;
        }

        double speedup = (serial_time > 0) ? serial_time / elapsed : 1.0;
        double efficiency = speedup / num_threads;

        printf("%-10d %-14.6f %-12.4f %-12.4f\n",
               num_threads, elapsed, speedup, efficiency);

        free(threads);
        free(tdata);
    }

    printf("===================================================================\n");

    free(image);
    free(output);

    return 0;
}

/*
 * =================================================================
 *  DISCUSSION QUESTIONS AND ANSWERS
 * =================================================================
 *
 * 1. How does increasing the number of threads affect execution time?
 *
 *    Increasing the number of threads generally decreases execution time
 *    because the workload (image rows) is divided among more threads that
 *    run concurrently on multiple CPU cores. Each thread processes fewer
 *    rows, so the total work finishes faster — up to a point. Once the
 *    number of threads exceeds the number of available physical cores,
 *    the benefit diminishes and may even reverse due to context-switching
 *    overhead.
 *
 * 2. Does speedup increase linearly with the number of threads? Explain.
 *
 *    No. Speedup does not increase linearly. While doubling threads might
 *    nearly halve execution time for small thread counts, the gains
 *    diminish as threads increase. This is due to:
 *    - Thread creation and joining overhead
 *    - Memory bandwidth saturation (all threads compete for the same
 *      memory bus)
 *    - Cache contention (threads may evict each other's cached data)
 *    - Load imbalance if rows are not perfectly divisible
 *    Amdahl's Law predicts that any serial fraction of the program
 *    fundamentally limits the maximum achievable speedup.
 *
 * 3. What factors limit the achievable speedup?
 *
 *    - Serial portions of the code (image initialization, output
 *      collection) that cannot be parallelized (Amdahl's Law)
 *    - Thread management overhead (creation, synchronization, joining)
 *    - Memory bandwidth: the Sobel filter is memory-bound; all threads
 *      read from the same image array, saturating the memory bus
 *    - Cache effects: false sharing, cache line bouncing, reduced
 *      effective cache per thread
 *    - Number of physical CPU cores: beyond this, threads must time-share
 *    - Load imbalance if rows are not evenly distributed
 *    - NUMA effects on multi-socket systems
 *
 * 4. Why can efficiency decrease as the number of threads increases?
 *
 *    Efficiency = Speedup / NumThreads. As threads increase:
 *    - The serial fraction becomes a larger relative bottleneck
 *    - Overhead per thread (creation, joining, scheduling) grows linearly
 *      while the per-thread workload shrinks
 *    - Memory bandwidth becomes the bottleneck, so adding threads does
 *      not proportionally increase throughput
 *    - More threads lead to more cache misses and contention
 *    Therefore the speedup grows sub-linearly, causing efficiency
 *    (speedup / threads) to drop.
 *
 * 5. Compare serial vs. Pthreads performance for edge detection.
 *
 *    For large images (e.g., 4096×4096), the Pthreads version with 4-8
 *    threads typically achieves 2-5× speedup over the serial version on
 *    a modern multi-core CPU. The edge detection problem is well-suited
 *    for parallelization because:
 *    - Each output pixel is independent (embarrassingly parallel)
 *    - No synchronization is needed between threads during computation
 *    - The workload can be evenly divided by rows
 *    However, the problem is memory-bound (the computation per pixel is
 *    light compared to the data movement), which limits the speedup
 *    compared to compute-bound workloads. The serial version is simpler
 *    and has no thread overhead, making it preferable for small images.
 *    For large images, the parallel version provides meaningful speedup.
 *
 * =================================================================
 */
