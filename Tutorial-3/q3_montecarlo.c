/*
 * Question 3: Monte Carlo Estimation of π using OpenMP
 * =====================================================
 * Method: Generate random points (x,y) in [0,1)×[0,1).
 *         Count how many fall inside the unit quarter-circle (x²+y² < 1).
 *         π ≈ 4 × (count inside) / (total points)
 *
 * Uses OpenMP reduction(+:count) for thread-safe accumulation.
 * Uses per-thread seeds for thread-safe random number generation.
 *
 * Compile: gcc -fopenmp -O2 -o q3_montecarlo q3_montecarlo.c -lm
 * Run:     ./q3_montecarlo
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
static inline int rand_r(unsigned int *seed) {
    *seed = *seed * 1103515245 + 12345;
    return (int)((unsigned int)(*seed / 65536) % 32768);
}
#endif

/*
 * Estimate π using Monte Carlo with n random samples.
 * Each thread gets its own seed based on thread ID for reproducibility
 * and thread safety (rand_r is thread-safe).
 */
double estimate_pi(long long n) {
    long long count = 0;

    #pragma omp parallel reduction(+:count)
    {
        /* Each thread has its own seed for thread-safe random generation */
        unsigned int seed = (unsigned int)(42 + omp_get_thread_num() * 17);

        #pragma omp for
        for (long long i = 0; i < n; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;

            /* Check if point (x,y) falls inside quarter circle of radius 1 */
            if (x * x + y * y <= 1.0) {
                count++;
            }
        }
    }

    return 4.0 * (double)count / (double)n;
}

int main() {
    /* Sample sizes to test */
    long long samples[] = {10LL, 100LL, 10000LL, 1000000LL,
                           10000000LL, 100000000LL, 1000000000LL};
    int num_tests = sizeof(samples) / sizeof(samples[0]);

    int num_threads = omp_get_max_threads();

    printf("============================================\n");
    printf("  Monte Carlo Estimation of Pi (OpenMP)\n");
    printf("  Threads: %d\n", num_threads);
    printf("============================================\n\n");

    printf("%-15s  %-18s  %-18s  %-12s\n",
           "Samples", "Pi Estimate", "Error", "Time (s)");
    printf("%-15s  %-18s  %-18s  %-12s\n",
           "---------------", "------------------", "------------------", "------------");

    for (int t = 0; t < num_tests; t++) {
        long long n = samples[t];

        double t_start = omp_get_wtime();
        double pi_est = estimate_pi(n);
        double t_end = omp_get_wtime();

        double error = fabs(pi_est - M_PI);

        printf("%-15lld  %-18.12f  %-18.12f  %-12.6f\n",
               n, pi_est, error, t_end - t_start);
    }

    printf("\n");
    printf("Reference value: Pi = %.15f\n", M_PI);
    printf("\nNote: Error decreases as O(1/sqrt(N)) — the Monte Carlo\n");
    printf("convergence rate. Doubling accuracy requires 4x the samples.\n");

    return 0;
}
