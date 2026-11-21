/*
 * =====================================================================================
 * Problem: Distributed Matrix Multiplication using MPI (Block-Cyclic Row Distribution)
 * File: matrix_mult_block_cyclic.c
 * Directory: MPI_Lab
 *
 * Concepts Demonstrated:
 *   1. Distributed Computing Paradigm:
 *      - Matrix A rows distributed in a block-cyclic fashion.
 *      - Matrix B broadcast to all processes (columns not distributed).
 *      - Each process independently computes assigned rows of C = A x B.
 *      - Partial results gathered back to Process 0.
 *   2. Target MPI Primitives:
 *      - MPI_Bcast()   : Broadcasts matrix B to all worker processes.
 *      - MPI_Scatter() : Scatters blocks of rows of A in cyclic rounds.
 *      - MPI_Gather()  : Gathers computed blocks of C back into the final matrix.
 * =====================================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Helper function to initialize matrix with deterministic values */
void initialize_matrices(int *A, int *B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            /* Use small values to prevent integer overflow during multiplication */
            A[i * N + j] = ((i + j) % 5) + 1;
            B[i * N + j] = ((i * 2 + j) % 4) + 1;
        }
    }
}

/* Helper function to print a small matrix or sub-matrix */
void print_matrix_sample(const char *name, int *M, int N, int max_display) {
    printf("Matrix %s (%d x %d):\n", name, N, N);
    int limit = (N < max_display) ? N : max_display;
    for (int i = 0; i < limit; i++) {
        printf("  [Row %3d]: ", i);
        for (int j = 0; j < limit; j++) {
            printf("%5d ", M[i * N + j]);
        }
        if (limit < N) printf("... ");
        printf("\n");
    }
    if (limit < N) {
        printf("  ... (%d more rows not displayed) ...\n", N - limit);
    }
    printf("\n");
}

/* Sequential matrix multiplication for verification */
int verify_result(int *A, int *B, int *C, int N) {
    int errors = 0;
    /* Verify a sample of rows (or all if N is small) to keep check fast */
    int sample_rows = (N > 100) ? 5 : N;
    for (int r = 0; r < sample_rows; r++) {
        int i = (r * (N / sample_rows));
        for (int j = 0; j < N; j++) {
            long long expected = 0;
            for (int k = 0; k < N; k++) {
                expected += (long long)A[i * N + k] * (long long)B[k * N + j];
            }
            if ((long long)C[i * N + j] != expected) {
                if (errors < 5) {
                    fprintf(stderr, "Mismatch at C[%d][%d]: Got %d, Expected %lld\n",
                            i, j, C[i * N + j], expected);
                }
                errors++;
            }
        }
    }
    return errors;
}

int main(int argc, char **argv) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Matrix Dimensions & Block Size:
     * - Default Block Size B = 500 (as specified in problem prompt)
     * - Default Dimension N = 1000 (total elements N x N = 10^6, configurable)
     */
    int block_size = 500;
    int N = 1000;

    if (argc >= 2) {
        N = atoi(argv[1]);
    }
    if (argc >= 3) {
        block_size = atoi(argv[2]);
    }

    /*
     * In Block-Cyclic distribution across 'size' processes:
     * One cycle/round distributes (size * block_size) rows.
     * Ensure N is a multiple of (size * block_size) for clean block-cyclic partitioning.
     */
    int rows_per_round = size * block_size;
    if (N % rows_per_round != 0) {
        int rounds = (N + rows_per_round - 1) / rows_per_round;
        int adjusted_N = rounds * rows_per_round;
        if (rank == 0) {
            printf("[Note] Adjusting N from %d to %d to be a multiple of (size * block_size = %d * %d = %d)\n",
                   N, adjusted_N, size, block_size, rows_per_round);
        }
        N = adjusted_N;
    }

    int num_rounds = N / rows_per_round;
    int local_rows = num_rounds * block_size;
    long long total_elements = (long long)N * N;

    if (rank == 0) {
        printf("================================================================================\n");
        printf(" MPI Distributed Matrix Multiplication (Block-Cyclic Row Distribution)\n");
        printf("================================================================================\n");
        printf(" Matrix Dimension (N x N)    : %d x %d\n", N, N);
        printf(" Total Matrix Elements (n)   : %lld (~10^%d elements)\n",
               total_elements, (total_elements >= 100000000 ? 8 : (total_elements >= 1000000 ? 6 : 4)));
        printf(" Number of MPI Processes     : %d\n", size);
        printf(" Block Size (rows per block) : %d rows\n", block_size);
        printf(" Number of Cyclic Rounds     : %d round(s)\n", num_rounds);
        printf(" Rows Assigned per Process   : %d rows (%d blocks)\n", local_rows, num_rounds);
        printf(" Communication Strategy      : 1D Row Block-Cyclic (MPI_Scatter, MPI_Bcast, MPI_Gather)\n");
        printf("================================================================================\n\n");
    }

    /* Matrices on Root Process (Process 0) */
    int *A = NULL;
    int *B = NULL;
    int *C = NULL;

    if (rank == 0) {
        A = (int *)malloc(total_elements * sizeof(int));
        B = (int *)malloc(total_elements * sizeof(int));
        C = (int *)malloc(total_elements * sizeof(int));

        if (!A || !B || !C) {
            fprintf(stderr, "Error: Process 0 failed to allocate memory for %d x %d matrices.\n", N, N);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        initialize_matrices(A, B, N);

        if (N <= 10) {
            print_matrix_sample("A (Initial)", A, N, 10);
            print_matrix_sample("B (Initial)", B, N, 10);
        } else {
            print_matrix_sample("A (Top-Left Sample)", A, N, 5);
            print_matrix_sample("B (Top-Left Sample)", B, N, 5);
        }
    }

    /*
     * Every process needs:
     * 1. A local buffer for its assigned rows of Matrix A: (local_rows * N) ints
     * 2. The entire Matrix B: (N * N) ints (since columns are not distributed)
     * 3. A local buffer for its computed rows of Matrix C: (local_rows * N) ints
     */
    int *local_A = (int *)malloc((long long)local_rows * N * sizeof(int));
    int *local_C = (int *)malloc((long long)local_rows * N * sizeof(int));
    int *full_B  = (rank == 0) ? B : (int *)malloc(total_elements * sizeof(int));

    if (!local_A || !local_C || !full_B) {
        fprintf(stderr, "Rank %d failed to allocate local buffers.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Synchronize before starting parallel benchmark */
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    /*
     * STEP 1: Broadcast Matrix B to all processes using MPI_Bcast()
     * Why? Each process calculates full dot-products of its assigned rows
     * with each column of B. Broadcasting B avoids complex column decomposition.
     */
    MPI_Bcast(full_B, (int)total_elements, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * STEP 2: Scatter rows of Matrix A in Block-Cyclic Fashion using MPI_Scatter()
     * In each cyclic round r:
     *   - Process 0 sends 1 block of (block_size * N) integers to Process 0, Process 1, ...
     *   - In round 0: Process 0 gets Block 0, Process 1 gets Block 1, etc.
     *   - In round 1: Process 0 gets Block P, Process 1 gets Block P+1, etc.
     */
    int elements_per_block = block_size * N;
    for (int r = 0; r < num_rounds; r++) {
        int *send_ptr = (rank == 0) ? &A[r * rows_per_round * N] : NULL;
        int *recv_ptr = &local_A[r * elements_per_block];

        MPI_Scatter(send_ptr, elements_per_block, MPI_INT,
                    recv_ptr, elements_per_block, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    /*
     * STEP 3: Independent Local Computation
     * Each process computes its independent assigned row values:
     *   local_C[row][col] = sum_{k=0}^{N-1} (local_A[row][k] * full_B[k][col])
     */
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            long long sum = 0;
            int a_row_offset = i * N;
            for (int k = 0; k < N; k++) {
                sum += (long long)local_A[a_row_offset + k] * (long long)full_B[k * N + j];
            }
            local_C[i * N + j] = (int)sum;
        }
    }

    /*
     * STEP 4: Gather Computed Rows back to Process 0 using MPI_Gather()
     * In each round r:
     *   - Process 0 gathers computed blocks from all processes and stores them
     *     contiguously into the final result matrix C.
     */
    for (int r = 0; r < num_rounds; r++) {
        int *send_ptr = &local_C[r * elements_per_block];
        int *recv_ptr = (rank == 0) ? &C[r * rows_per_round * N] : NULL;

        MPI_Gather(send_ptr, elements_per_block, MPI_INT,
                   recv_ptr, elements_per_block, MPI_INT,
                   0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    /*
     * STEP 5: Process 0 Displays the Final Matrix Results and Verifies Correctness
     */
    if (rank == 0) {
        printf("--------------------------------------------------------------------------------\n");
        printf(" Computation & Communication Completed Successfully!\n");
        printf(" Total Elapsed Parallel Time: %.4f seconds\n", elapsed_time);
        double gflops = (2.0 * (double)N * (double)N * (double)N) / (elapsed_time * 1e9);
        printf(" Estimated Performance      : %.2f GFLOP/s\n", gflops);
        printf("--------------------------------------------------------------------------------\n\n");

        if (N <= 10) {
            print_matrix_sample("C (Final Result Matrix)", C, N, 10);
        } else {
            print_matrix_sample("C (Top-Left 5x5 Corner of Result Matrix)", C, N, 5);
        }

        /* Verify correctness */
        printf("Verifying distributed computation against sequential reference...\n");
        int errors = verify_result(A, B, C, N);
        if (errors == 0) {
            printf("[VERIFICATION PASSED] Result matrix C matches exact mathematical multiplication!\n");
        } else {
            printf("[VERIFICATION FAILED] Found %d numerical discrepancies!\n", errors);
        }
        printf("================================================================================\n");

        free(A);
        free(B);
        free(C);
    } else {
        free(full_B);
    }

    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}

