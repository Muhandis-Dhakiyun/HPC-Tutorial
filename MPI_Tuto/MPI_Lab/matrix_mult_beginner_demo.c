/*
 * =====================================================================================
 * Problem: Beginner's Step-by-Step MPI Matrix Multiplication
 * File: matrix_mult_beginner_demo.c
 * Directory: MPI_Lab
 *
 * Designed for someone who has just started MPI programming:
 *   - Uses a small 8x8 matrix with Block Size = 2 rows
 *   - Prints every step explicitly:
 *       1. Process 0 creates matrices A and B.
 *       2. Process 0 broadcasts matrix B using MPI_Bcast().
 *       3. Process 0 scatters rows of A in blocks using MPI_Scatter() (Block-Cyclic).
 *       4. Each process prints the exact rows it received.
 *       5. Each process independently computes its rows of C.
 *       6. Process 0 gathers all computed rows back using MPI_Gather().
 *       7. Process 0 displays the complete final result matrix C.
 * =====================================================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8          /* Matrix size: 8x8 (64 elements) */
#define BLOCK_SIZE 2 /* Block size: 2 rows per block */

void print_full_matrix(const char *title, int M[N][N]) {
    printf("--- %s ---\n", title);
    for (int i = 0; i < N; i++) {
        printf("Row %d: [ ", i);
        for (int j = 0; j < N; j++) {
            printf("%4d ", M[i][j]);
        }
        printf("]\n");
    }
    printf("\n");
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2 && size != 4) {
        if (rank == 0) {
            printf("[Notice] Please run this demo with 2 or 4 processes for clean 8x8 division.\n");
            printf("         Example: mpiexec -n 2 .\\matrix_mult_beginner_demo.exe\n");
        }
    }

    int rows_per_round = size * BLOCK_SIZE;
    int num_rounds = N / rows_per_round;
    int local_rows = num_rounds * BLOCK_SIZE;

    int A[N][N];
    int B[N][N];
    int C[N][N];

    /* Process 0 generates the initial matrices */
    if (rank == 0) {
        printf("======================================================================\n");
        printf(" BEGINNER'S STEP-BY-STEP MPI MATRIX MULTIPLICATION DEMO\n");
        printf(" Matrix Size: %dx%d | Processes: %d | Block Size: %d rows\n", N, N, size, BLOCK_SIZE);
        printf("======================================================================\n\n");

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] = i + 1;       /* Row i contains all (i + 1) */
                B[i][j] = (i == j) ? 2 : 1; /* Diagonal is 2, rest is 1 */
            }
        }

        print_full_matrix("Process 0 Generated Matrix A", A);
        print_full_matrix("Process 0 Generated Matrix B", B);

        printf("----------------------------------------------------------------------\n");
        printf(" STEP 1: Broadcasting Matrix B to ALL processes via MPI_Bcast()...\n");
        printf("----------------------------------------------------------------------\n");
    }

    /* STEP 1: Process 0 broadcasts matrix B to all processes */
    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank != 0) {
        printf("[Process %d] Successfully received full Matrix B via MPI_Bcast()!\n", rank);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Local buffers for each process */
    int local_A[local_rows][N];
    int local_C[local_rows][N];

    if (rank == 0) {
        printf("\n----------------------------------------------------------------------\n");
        printf(" STEP 2: Distributing Matrix A rows in BLOCK-CYCLIC fashion via MPI_Scatter()...\n");
        printf("         Each process gets %d rows per round across %d round(s).\n", BLOCK_SIZE, num_rounds);
        printf("----------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* STEP 2: Scatter rows in block-cyclic rounds */
    int elements_per_block = BLOCK_SIZE * N;
    for (int r = 0; r < num_rounds; r++) {
        int *send_ptr = (rank == 0) ? &A[r * rows_per_round][0] : NULL;
        int *recv_ptr = &local_A[r * BLOCK_SIZE][0];

        MPI_Scatter(send_ptr, elements_per_block, MPI_INT,
                    recv_ptr, elements_per_block, MPI_INT,
                    0, MPI_COMM_WORLD);

        /* Print which rows this process received in this round */
        int global_start_row = (r * size + rank) * BLOCK_SIZE;
        printf("[Process %d] Round %d: Received Block (Global Rows %d and %d)\n",
               rank, r, global_start_row, global_start_row + 1);
        fflush(stdout);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* STEP 3: Each process independently computes its assigned rows */
    if (rank == 0) {
        printf("\n----------------------------------------------------------------------\n");
        printf(" STEP 3: Each process independently computes its assigned rows (C = A x B)...\n");
        printf("----------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += local_A[i][k] * B[k][j];
            }
            local_C[i][j] = sum;
        }
    }

    /* Print a sample computed row from each process */
    printf("[Process %d] Finished computing its %d assigned rows! Sample local row 0 result: [%d, %d, ...]\n",
           rank, local_rows, local_C[0][0], local_C[0][1]);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);

    /* STEP 4: Gather computed blocks of C back into Process 0 */
    if (rank == 0) {
        printf("\n----------------------------------------------------------------------\n");
        printf(" STEP 4: Process 0 gathers computed blocks from all processes via MPI_Gather()...\n");
        printf("----------------------------------------------------------------------\n");
    }

    for (int r = 0; r < num_rounds; r++) {
        int *send_ptr = &local_C[r * BLOCK_SIZE][0];
        int *recv_ptr = (rank == 0) ? &C[r * rows_per_round][0] : NULL;

        MPI_Gather(send_ptr, elements_per_block, MPI_INT,
                   recv_ptr, elements_per_block, MPI_INT,
                   0, MPI_COMM_WORLD);
    }

    /* STEP 5: Process 0 displays the final matrix result */
    if (rank == 0) {
        printf("\n======================================================================\n");
        printf(" STEP 5: Process 0 Displays the Final Gathered Matrix C Result\n");
        printf("======================================================================\n");
        print_full_matrix("Final Gathered Matrix C = A x B", C);

        /* Simple verification check for row 0:
         * A[0][k] = 1 for all k in 0..7.
         * B diagonal = 2, others = 1.
         * Sum over k = 1*(2) + 7*(1) = 9. Every element of row 0 should be 9!
         */
        int ok = 1;
        for (int j = 0; j < N; j++) {
            if (C[0][j] != 9) ok = 0;
        }
        if (ok) {
            printf("[VERIFICATION SUCCESS] Row 0 values are all 9 as mathematically expected!\n");
        }
        printf("======================================================================\n");
    }

    MPI_Finalize();
    return 0;
}

