/*
 * Problem 3: Non-Blocking Communication and Overlap
 *
 * Objective:
 *   Replace MPI_Send/MPI_Recv with MPI_Isend/MPI_Irecv and investigate
 *   whether independent computation can overlap with communication.
 *   Measures 1 MB, 10 MB, and 50 MB messages and compares blocking vs non-blocking time.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_SIZES 3
#define NUM_TRIALS 3

/* Structure to define test cases */
typedef struct {
    const char *label;
    long count; /* Number of 4-byte integers */
    double size_mb;
    long comp_workload; /* Computation loop iterations */
} TestCase;

/* Independent computation that does not depend on communicated buffers */
volatile double dummy_sink = 0.0;
void do_dummy_computation(long iterations) {
    double acc = 0.0;
    for (long i = 0; i < iterations; i++) {
        acc += (i & 1) ? 0.0001 * (i % 100) : -0.00005 * (i % 50);
    }
    dummy_sink = acc;
}

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

    TestCase test_cases[NUM_SIZES] = {
        {"1 MB",  250000L,   1.0,  5000000L},
        {"10 MB", 2500000L,  10.0, 15000000L},
        {"50 MB", 12500000L, 50.0, 40000000L}
    };

    double avg_blocking[NUM_SIZES];
    double avg_nonblocking[NUM_SIZES];
    double avg_pure_comm[NUM_SIZES];
    double avg_pure_comp[NUM_SIZES];

    if (rank == 0) {
        printf("========================================================================\n");
        printf(" Problem 3: Non-Blocking Communication & Computation Overlap Investigation\n");
        printf("========================================================================\n");
    }

    for (int s = 0; s < NUM_SIZES; s++) {
        long count = test_cases[s].count;
        long workload = test_cases[s].comp_workload;

        int *send_buf = NULL;
        int *recv_buf = NULL;

        if (rank == 0) {
            send_buf = (int *)malloc(count * sizeof(int));
            for (long i = 0; i < count; i++) {
                send_buf[i] = (int)(i + 1);
            }
        } else if (rank == 1) {
            recv_buf = (int *)malloc(count * sizeof(int));
        }

        double sum_pure_comp = 0.0;
        double sum_pure_comm = 0.0;
        double sum_blocking = 0.0;
        double sum_nonblocking = 0.0;

        for (int trial = 0; trial < NUM_TRIALS; trial++) {
            /* 1. Measure pure computation time */
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            do_dummy_computation(workload);
            MPI_Barrier(MPI_COMM_WORLD);
            double t_comp = MPI_Wtime() - t0;
            sum_pure_comp += t_comp;

            /* 2. Measure pure blocking communication time */
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            if (rank == 0) {
                MPI_Send(send_buf, (int)count, MPI_INT, 1, 10, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Status st;
                MPI_Recv(recv_buf, (int)count, MPI_INT, 0, 10, MPI_COMM_WORLD, &st);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            double t_comm = MPI_Wtime() - t0;
            sum_pure_comm += t_comm;

            /* 3. Sequential: Blocking Communication followed by Computation */
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            if (rank == 0) {
                MPI_Send(send_buf, (int)count, MPI_INT, 1, 20, MPI_COMM_WORLD);
            } else if (rank == 1) {
                MPI_Status st;
                MPI_Recv(recv_buf, (int)count, MPI_INT, 0, 20, MPI_COMM_WORLD, &st);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            do_dummy_computation(workload);
            MPI_Barrier(MPI_COMM_WORLD);
            double t_block_total = MPI_Wtime() - t0;
            sum_blocking += t_block_total;

            /* 4. Non-Blocking: Initiate Isend/Irecv, perform computation, then MPI_Wait */
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            MPI_Request req;
            if (rank == 0) {
                /* Initiate send */
                MPI_Isend(send_buf, (int)count, MPI_INT, 1, 30, MPI_COMM_WORLD, &req);
            } else if (rank == 1) {
                /* Initiate receive */
                MPI_Irecv(recv_buf, (int)count, MPI_INT, 0, 30, MPI_COMM_WORLD, &req);
            }

            /* Perform independent dummy computation during transfer */
            do_dummy_computation(workload);

            /* Wait for request completion before reusing/accessing buffers */
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            MPI_Barrier(MPI_COMM_WORLD);
            double t_nonblock_total = MPI_Wtime() - t0;
            sum_nonblocking += t_nonblock_total;

            /* Verification on process 1 */
            if (trial == 0 && rank == 1) {
                printf("[Verification %s] Process 1 non-blocking received: first = %d, last = %d\n",
                       test_cases[s].label, recv_buf[0], recv_buf[count - 1]);
            }
        }

        if (rank == 0) {
            avg_pure_comp[s] = sum_pure_comp / NUM_TRIALS;
            avg_pure_comm[s] = sum_pure_comm / NUM_TRIALS;
            avg_blocking[s] = sum_blocking / NUM_TRIALS;
            avg_nonblocking[s] = sum_nonblocking / NUM_TRIALS;
        }

        if (rank == 0 && send_buf) free(send_buf);
        if (rank == 1 && recv_buf) free(recv_buf);
    }

    if (rank == 0) {
        printf("\n--------------------------------------------------------------------------------------------------------\n");
        printf(" Timing Comparison Table: Blocking (Comm + Comp) vs Non-Blocking (Overlapped)\n");
        printf("--------------------------------------------------------------------------------------------------------\n");
        printf("%-10s %-16s %-16s %-18s %-18s %-12s\n",
               "Size", "Comm Only (s)", "Comp Only (s)", "Sequential (s)", "Overlapped (s)", "Speedup/Overlap");
        printf("--------------------------------------------------------------------------------------------------------\n");
        for (int s = 0; s < NUM_SIZES; s++) {
            double diff = avg_blocking[s] - avg_nonblocking[s];
            double pct = (diff / avg_blocking[s]) * 100.0;
            printf("%-10s %-16.6f %-16.6f %-18.6f %-18.6f %+.2f%%\n",
                   test_cases[s].label,
                   avg_pure_comm[s],
                   avg_pure_comp[s],
                   avg_blocking[s],
                   avg_nonblocking[s],
                   pct);
        }
        printf("--------------------------------------------------------------------------------------------------------\n");

        /* Save to CSV */
        FILE *fp = fopen("nonblocking_comparison.csv", "w");
        if (fp) {
            fprintf(fp, "Size,Comm_Only_s,Comp_Only_s,Blocking_Total_s,NonBlocking_Total_s\n");
            for (int s = 0; s < NUM_SIZES; s++) {
                fprintf(fp, "%s,%.6f,%.6f,%.6f,%.6f\n",
                        test_cases[s].label,
                        avg_pure_comm[s],
                        avg_pure_comp[s],
                        avg_blocking[s],
                        avg_nonblocking[s]);
            }
            fclose(fp);
            printf("Saved comparison results to 'nonblocking_comparison.csv'.\n");
        }
    }

    MPI_Finalize();
    return 0;
}

