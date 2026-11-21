/*
 * Problem 2: Blocking Point-to-Point Communication
 *
 * Objective:
 *   Send an integer array from Process 0 to Process 1 using MPI_Send and MPI_Recv,
 *   then measure elapsed time using MPI_Wtime().
 *   Evaluates message sizes of 1 MB, 10 MB, and 50 MB across multiple trials.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_TRIALS 3
#define NUM_SIZES 3

/* Structure to define test cases */
typedef struct {
    const char *label;
    long count; /* Number of 4-byte integers */
    double size_mb;
} TestCase;

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Error: This program requires at least 2 processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /* Test configurations based on Section 6.4:
     * 1 MB  = 250,000 ints
     * 10 MB = 2,500,000 ints
     * 50 MB = 12,500,000 ints
     */
    TestCase test_cases[NUM_SIZES] = {
        {"1 MB",  250000L,   1.0},
        {"10 MB", 2500000L,  10.0},
        {"50 MB", 12500000L, 50.0}
    };

    double results[NUM_SIZES][NUM_TRIALS];
    double averages[NUM_SIZES];

    if (rank == 0) {
        printf("========================================================================\n");
        printf(" Problem 2: Blocking Point-to-Point Communication (Process 0 -> Process 1)\n");
        printf("========================================================================\n");
    }

    for (int s = 0; s < NUM_SIZES; s++) {
        long count = test_cases[s].count;
        double sum_time = 0.0;

        int *send_buf = NULL;
        int *recv_buf = NULL;

        if (rank == 0) {
            send_buf = (int *)malloc(count * sizeof(int));
            if (!send_buf) {
                fprintf(stderr, "Process 0 failed to allocate memory for %s\n", test_cases[s].label);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            /* Initialize array data */
            for (long i = 0; i < count; i++) {
                send_buf[i] = (int)(i + 1);
            }
        } else if (rank == 1) {
            recv_buf = (int *)malloc(count * sizeof(int));
            if (!recv_buf) {
                fprintf(stderr, "Process 1 failed to allocate memory for %s\n", test_cases[s].label);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            /* Synchronize before timing */
            MPI_Barrier(MPI_COMM_WORLD);
            double start = MPI_Wtime();

            if (rank == 0) {
                /* Blocking send to rank 1 */
                MPI_Send(send_buf, (int)count, MPI_INT, 1, 0, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Status status;
                /* Blocking receive from rank 0 */
                MPI_Recv(recv_buf, (int)count, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

                /* Print verification for the first trial */
                if (trial == 0) {
                    printf("[Verification %s] Process 1 received array: first = %d, last = %d\n",
                           test_cases[s].label, recv_buf[0], recv_buf[count - 1]);
                }
            }

            /* Synchronize after communication to ensure receiver completion */
            MPI_Barrier(MPI_COMM_WORLD);
            double elapsed = MPI_Wtime() - start;

            if (rank == 0) {
                results[s][trial] = elapsed;
                sum_time += elapsed;
            }
        }

        if (rank == 0) {
            averages[s] = sum_time / NUM_TRIALS;
        }

        if (rank == 0 && send_buf) free(send_buf);
        if (rank == 1 && recv_buf) free(recv_buf);
    }

    if (rank == 0) {
        printf("\n------------------------------------------------------------------------\n");
        printf(" Timing Results Summary (Blocking Point-to-Point Communication)\n");
        printf("------------------------------------------------------------------------\n");
        printf("%-12s %-14s %-14s %-14s %-14s %-14s\n",
               "Message Size", "Trial 1 (s)", "Trial 2 (s)", "Trial 3 (s)", "Average (s)", "Bandwidth (MB/s)");
        printf("------------------------------------------------------------------------\n");
        for (int s = 0; s < NUM_SIZES; s++) {
            double bw = test_cases[s].size_mb / averages[s];
            printf("%-12s %-14.6f %-14.6f %-14.6f %-14.6f %-14.2f\n",
                   test_cases[s].label,
                   results[s][0], results[s][1], results[s][2],
                   averages[s], bw);
        }
        printf("------------------------------------------------------------------------\n");

        /* Save results to CSV file for plotting */
        FILE *fp = fopen("blocking_results.csv", "w");
        if (fp) {
            fprintf(fp, "Size_MB,Trial1_s,Trial2_s,Trial3_s,Average_s,Bandwidth_MBs\n");
            for (int s = 0; s < NUM_SIZES; s++) {
                double bw = test_cases[s].size_mb / averages[s];
                fprintf(fp, "%.1f,%.6f,%.6f,%.6f,%.6f,%.2f\n",
                        test_cases[s].size_mb,
                        results[s][0], results[s][1], results[s][2],
                        averages[s], bw);
            }
            fclose(fp);
            printf("Saved timing results to 'blocking_results.csv'.\n");
        }
    }

    MPI_Finalize();
    return 0;
}

