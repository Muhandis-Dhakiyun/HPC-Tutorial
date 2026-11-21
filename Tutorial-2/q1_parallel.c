/*
 * q1_parallel.c - Parallel Monte Carlo Estimation of π Using Pthreads
 *
 * Method: Divide N random points among T threads. Each thread uses rand_r()
 *         with a unique per-thread seed for thread safety. Partial counts are
 *         summed after all threads join.
 *
 * Compile: gcc -o q1_parallel q1_parallel.c -lpthread -lm
 * Run:     ./q1_parallel
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
static inline int rand_r(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((unsigned int)(*seed / 65536) % 32768);
}
#endif

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

/* Shared data passed to each thread */
typedef struct {
    long num_points;    /* Number of points this thread generates */
    long count;         /* Number of points inside the quarter-circle */
    unsigned int seed;  /* Per-thread seed for rand_r() */
} thread_data_t;

/* Thread function: generate random points and count hits */
void *monte_carlo(void *arg) {
    thread_data_t *data = (thread_data_t *)arg;
    long local_count = 0;
    unsigned int seed = data->seed;

    for (long i = 0; i < data->num_points; i++) {
        double x = (double)rand_r(&seed) / RAND_MAX;
        double y = (double)rand_r(&seed) / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    data->count = local_count;
    return NULL;
}

int main(void) {
    /* Sample sizes to test */
    long sample_sizes[] = {10, 100, 10000, 1000000, 10000000, 100000000, 1000000000L};
    int num_sizes = sizeof(sample_sizes) / sizeof(sample_sizes[0]);

    /* Thread counts to test */
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_thread_counts = sizeof(thread_counts) / sizeof(thread_counts[0]);

    printf("===========================================================================================\n");
    printf("     Parallel Monte Carlo Estimation of Pi (Pthreads)\n");
    printf("===========================================================================================\n");

    for (int s = 0; s < num_sizes; s++) {
        long N = sample_sizes[s];
        double serial_time = -1.0; /* Will store time for 1 thread (baseline) */

        printf("\n--- N = %ld ---\n", N);
        printf("%-10s %-14s %-14s %-14s %-12s %-12s\n",
               "Threads", "Pi Estimate", "Error", "Time (s)", "Speedup", "Efficiency");
        printf("-------------------------------------------------------------------------------------------\n");

        for (int t = 0; t < num_thread_counts; t++) {
            int num_threads = thread_counts[t];
            pthread_t *threads = malloc(num_threads * sizeof(pthread_t));
            thread_data_t *tdata = malloc(num_threads * sizeof(thread_data_t));

            /* Divide work among threads */
            long points_per_thread = N / num_threads;
            long remainder = N % num_threads;

            for (int i = 0; i < num_threads; i++) {
                /* Distribute remainder points to first 'remainder' threads */
                tdata[i].num_points = points_per_thread + (i < remainder ? 1 : 0);
                tdata[i].count = 0;
                tdata[i].seed = 42 + i; /* Unique seed per thread */
            }

            double start = get_time();

            /* Create threads */
            for (int i = 0; i < num_threads; i++) {
                pthread_create(&threads[i], NULL, monte_carlo, &tdata[i]);
            }

            /* Wait for all threads to finish */
            for (int i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
            }

            double end = get_time();
            double elapsed = end - start;

            /* Sum partial counts */
            long total_count = 0;
            for (int i = 0; i < num_threads; i++) {
                total_count += tdata[i].count;
            }

            double pi_estimate = 4.0 * (double)total_count / (double)N;
            double error = fabs(pi_estimate - M_PI);

            /* Compute speedup and efficiency relative to 1-thread run */
            if (num_threads == 1) {
                serial_time = elapsed;
            }

            double speedup = (serial_time > 0) ? serial_time / elapsed : 1.0;
            double efficiency = speedup / num_threads;

            printf("%-10d %-14.10f %-14.10f %-14.6f %-12.4f %-12.4f\n",
                   num_threads, pi_estimate, error, elapsed, speedup, efficiency);

            free(threads);
            free(tdata);
        }
    }

    printf("\n===========================================================================================\n");
    printf("Reference Pi = %.15f\n", M_PI);

    return 0;
}
