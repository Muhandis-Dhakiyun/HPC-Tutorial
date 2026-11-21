/*
 * Question 1: Analysis of Three OpenMP Code Variants
 * ===================================================
 * This file contains three code variants that parallelize the update
 *     v[indices[i]] += f(i);
 * and explains the correctness and performance trade-offs of each.
 *
 * Compile: gcc -fopenmp -o q1_analysis q1_analysis.c -lm
 * Run:     ./q1_analysis
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* A simple function that takes some time */
double f(int i) {
    return sin((double)i) * sin((double)i);
}

#define N 1000
#define V_SIZE 100

/*
 * ============================================================
 * CODE VARIANT 1: Race Condition (INCORRECT)
 * ============================================================
 *
 * #pragma omp parallel for default(none) shared(v, indices, n)
 * for (i = 0; i < n; i++)
 *     v[indices[i]] += f(i);
 *
 * PROBLEM: Multiple threads may simultaneously read-modify-write
 * the same v[indices[i]] when two different values of i map to
 * the same index via indices[]. This is a classic race condition.
 * The result is non-deterministic and incorrect.
 *
 * Example: If indices[3] == indices[7] == 5, then thread A and
 * thread B may both read v[5], compute f(3) and f(7) respectively,
 * and write back — one update is lost.
 */
void variant1_race_condition(double *v, int *indices, int n) {
    int i;
    printf("Variant 1 (Race Condition):\n");

    /* Reset v */
    for (i = 0; i < V_SIZE; i++) v[i] = 0.0;

    #pragma omp parallel for default(none) shared(v, indices, n)
    for (i = 0; i < n; i++) {
        v[indices[i]] += f(i);  /* RACE CONDITION: multiple threads may update same v[indices[i]] */
    }

    double sum = 0.0;
    for (i = 0; i < V_SIZE; i++) sum += v[i];
    printf("  Sum of v[] = %f (may vary between runs due to race condition)\n", sum);
}

/*
 * ============================================================
 * CODE VARIANT 2: Critical Section (CORRECT but SLOW)
 * ============================================================
 *
 * #pragma omp parallel for default(none) shared(v, indices, n)
 * for (i = 0; i < n; i++) {
 *     #pragma omp critical
 *     v[indices[i]] += f(i);
 * }
 *
 * CORRECTNESS: The critical section ensures that only one thread
 * at a time executes the update, eliminating the race condition.
 *
 * PERFORMANCE: This effectively serializes ALL updates — even when
 * two threads update DIFFERENT elements of v[]. This loses almost
 * all parallelism benefit. The overhead of acquiring/releasing
 * the critical section lock on every iteration makes this often
 * SLOWER than the serial version.
 */
void variant2_critical(double *v, int *indices, int n) {
    int i;
    printf("Variant 2 (Critical Section):\n");

    for (i = 0; i < V_SIZE; i++) v[i] = 0.0;

    double t_start = omp_get_wtime();

    #pragma omp parallel for default(none) shared(v, indices, n)
    for (i = 0; i < n; i++) {
        #pragma omp critical
        {
            v[indices[i]] += f(i);  /* Correct but serialized */
        }
    }

    double t_end = omp_get_wtime();
    double sum = 0.0;
    for (i = 0; i < V_SIZE; i++) sum += v[i];
    printf("  Sum of v[] = %f  Time = %f s\n", sum, t_end - t_start);
}

/*
 * ============================================================
 * CODE VARIANT 3: Atomic Update (CORRECT and FASTER)
 * ============================================================
 *
 * #pragma omp parallel for default(none) shared(v, indices, n)
 * for (i = 0; i < n; i++) {
 *     #pragma omp atomic
 *     v[indices[i]] += f(i);
 * }
 *
 * CORRECTNESS: The atomic directive ensures the read-modify-write
 * of v[indices[i]] is performed atomically — no lost updates.
 *
 * PERFORMANCE: Unlike critical, atomic allows DIFFERENT elements
 * of v[] to be updated concurrently. Only updates to the SAME
 * memory location are serialized. This is typically implemented
 * using hardware atomic instructions (e.g., compare-and-swap),
 * which have much lower overhead than a mutex.
 *
 * This is the best solution when the update is a simple operation
 * (+=, -=, *=, etc.) supported by #pragma omp atomic.
 *
 * NOTE: If we could restructure the problem to avoid index
 * conflicts entirely (e.g., using reduction or private arrays
 * that are merged), that could be even better, but atomic is
 * the most practical fix for indirect indexing patterns.
 */
void variant3_atomic(double *v, int *indices, int n) {
    int i;
    printf("Variant 3 (Atomic):\n");

    for (i = 0; i < V_SIZE; i++) v[i] = 0.0;

    double t_start = omp_get_wtime();

    #pragma omp parallel for default(none) shared(v, indices, n)
    for (i = 0; i < n; i++) {
        #pragma omp atomic
        v[indices[i]] += f(i);  /* Correct and allows concurrent updates to different elements */
    }

    double t_end = omp_get_wtime();
    double sum = 0.0;
    for (i = 0; i < V_SIZE; i++) sum += v[i];
    printf("  Sum of v[] = %f  Time = %f s\n", sum, t_end - t_start);
}

/* Serial reference for correctness check */
void serial_reference(double *v, int *indices, int n) {
    int i;
    printf("Serial Reference:\n");

    for (i = 0; i < V_SIZE; i++) v[i] = 0.0;

    for (i = 0; i < n; i++) {
        v[indices[i]] += f(i);
    }

    double sum = 0.0;
    for (i = 0; i < V_SIZE; i++) sum += v[i];
    printf("  Sum of v[] = %f (correct answer)\n", sum);
}

int main() {
    double v[V_SIZE];
    int indices[N];
    int i;

    /* Initialize indices to create conflicts (many map to same v element) */
    srand(42);
    for (i = 0; i < N; i++) {
        indices[i] = rand() % V_SIZE;
    }

    printf("=== Question 1: OpenMP Code Variant Analysis ===\n");
    printf("N = %d, V_SIZE = %d, Threads = %d\n\n", N, V_SIZE, omp_get_max_threads());

    serial_reference(v, indices, N);
    printf("\n");
    variant1_race_condition(v, indices, N);
    printf("\n");
    variant2_critical(v, indices, N);
    printf("\n");
    variant3_atomic(v, indices, N);

    printf("\n=== Summary ===\n");
    printf("Variant 1 (no protection):  INCORRECT — race condition\n");
    printf("Variant 2 (critical):       CORRECT   — but serializes all updates\n");
    printf("Variant 3 (atomic):         CORRECT   — best performance, allows concurrency\n");

    return 0;
}
