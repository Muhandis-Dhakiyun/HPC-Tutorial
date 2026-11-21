/*
 * Question 6: Determine Default OpenMP Scheduling Policy
 * ========================================================
 * This program determines whether the default schedule for
 * #pragma omp for is static, dynamic, or guided by observing
 * which thread gets which iteration.
 *
 * Method:
 * - Run a parallel for loop with the default schedule.
 * - Record which thread executes each iteration.
 * - Analyze the pattern:
 *   - STATIC:  Contiguous blocks of roughly equal size
 *   - DYNAMIC: Interleaved, non-deterministic assignment
 *   - GUIDED:  Decreasing chunk sizes
 *
 * Compile: gcc -fopenmp -O2 -o q6_schedule q6_schedule.c
 * Run:     ./q6_schedule
 *          OMP_SCHEDULE="dynamic,1" ./q6_schedule   (to test detection)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NUM_ITERATIONS 64

int main() {
    int thread_map[NUM_ITERATIONS];  /* Records which thread got which iteration */
    int num_threads;

    printf("============================================\n");
    printf("  Determine Default OpenMP Schedule\n");
    printf("  Iterations: %d\n", NUM_ITERATIONS);
    printf("============================================\n\n");

    /* Execute parallel for with DEFAULT schedule */
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();

        #pragma omp for schedule(runtime)
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            thread_map[i] = omp_get_thread_num();
        }
    }

    printf("Number of threads: %d\n\n", num_threads);

    /* Print the iteration-to-thread mapping */
    printf("Iteration -> Thread mapping:\n");
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        printf("  iter %2d -> thread %d", i, thread_map[i]);
        if ((i + 1) % 4 == 0) printf("\n");
        else printf("   ");
    }
    printf("\n\n");

    /* Analyze the pattern */
    printf("Per-thread iteration ranges:\n");
    for (int t = 0; t < num_threads; t++) {
        printf("  Thread %d: ", t);
        int count = 0;
        int first = -1, last = -1;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            if (thread_map[i] == t) {
                if (first == -1) first = i;
                last = i;
                count++;
            }
        }
        printf("iterations %d to %d (%d total)\n", first, last, count);
    }
    printf("\n");

    /*
     * Detect scheduling type by analyzing the pattern:
     * 1. Check if each thread's iterations form a contiguous block → STATIC
     * 2. Check if chunk sizes decrease → GUIDED
     * 3. Otherwise → DYNAMIC
     */

    /* Check for static: each thread has contiguous iterations */
    int is_static = 1;
    for (int t = 0; t < num_threads && is_static; t++) {
        int in_block = 0;
        int block_ended = 0;
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            if (thread_map[i] == t) {
                if (block_ended) {
                    is_static = 0;  /* Found a gap — not contiguous */
                    break;
                }
                in_block = 1;
            } else if (in_block) {
                block_ended = 1;
                in_block = 0;
            }
        }
    }

    /* Check for guided: chunks assigned to threads decrease in size */
    int is_guided = 0;
    if (!is_static) {
        /* Count consecutive runs (chunks) and check if sizes decrease */
        int chunks[NUM_ITERATIONS];
        int num_chunks = 0;
        int i = 0;
        while (i < NUM_ITERATIONS) {
            int t = thread_map[i];
            int chunk_size = 0;
            while (i < NUM_ITERATIONS && thread_map[i] == t) {
                chunk_size++;
                i++;
            }
            chunks[num_chunks++] = chunk_size;
        }

        /* Guided: first chunks are large, later chunks are smaller */
        if (num_chunks >= 4) {
            int decreasing = 1;
            /* Check if the first half has larger average than second half */
            int first_half_sum = 0, second_half_sum = 0;
            int mid = num_chunks / 2;
            for (int c = 0; c < mid; c++) first_half_sum += chunks[c];
            for (int c = mid; c < num_chunks; c++) second_half_sum += chunks[c];
            if (first_half_sum > second_half_sum * 1.5) {
                is_guided = 1;
            }
        }
    }

    /* Report result */
    printf("Schedule Detection Result:\n");
    if (is_static) {
        printf("  >>> DEFAULT SCHEDULE IS: STATIC <<<\n");
        printf("  Evidence: Each thread has a contiguous block of iterations.\n");
        printf("  With %d iterations and %d threads, each gets ~%d iterations.\n",
               NUM_ITERATIONS, num_threads, NUM_ITERATIONS / num_threads);
    } else if (is_guided) {
        printf("  >>> DEFAULT SCHEDULE IS: GUIDED <<<\n");
        printf("  Evidence: Chunk sizes decrease over the iteration range.\n");
        printf("  Early chunks are large, later chunks are small (minimum 1).\n");
    } else {
        printf("  >>> DEFAULT SCHEDULE IS: DYNAMIC <<<\n");
        printf("  Evidence: Non-contiguous, interleaved iteration assignment.\n");
        printf("  Iterations are assigned to threads on-demand.\n");
    }

    printf("\nNote: The OpenMP standard says the default schedule is\n");
    printf("implementation-defined. Most implementations (GCC, Clang, ICC)\n");
    printf("use static scheduling by default. The OMP_SCHEDULE environment\n");
    printf("variable controls schedule(runtime).\n");

    return 0;
}
