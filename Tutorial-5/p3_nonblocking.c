/*
 * Problem 3: Non-Blocking Communication
 *
 * Replaces MPI_Send/MPI_Recv with MPI_Isend/MPI_Irecv. While communication
 * is in progress, a dummy computation is performed to demonstrate overlap.
 * MPI_Wait is called before accessing received data.
 *
 * Same message sizes as Problem 2: 1 MB, 10 MB, 50 MB.
 * 3 trials per size, reports average time.
 *
 * Compile: mpicc -o p3_nonblocking p3_nonblocking.c -lm
 * Run:     mpirun -np 2 ./p3_nonblocking
 *
 * DISCUSSION — OVERLAP OF COMPUTATION AND COMMUNICATION:
 *
 * Non-blocking calls (MPI_Isend / MPI_Irecv) return immediately, allowing the
 * process to perform useful computation while the data transfer proceeds in
 * the background. MPI_Wait then blocks until the operation completes.
 *
 * In an ideal scenario, if the computation takes as long as the communication,
 * the total time is max(T_comp, T_comm) instead of T_comp + T_comm, yielding
 * significant speedup. In practice, the degree of overlap depends on:
 *
 *   - Whether the MPI implementation truly progresses the transfer in the
 *     background (some implementations defer work until MPI_Wait/MPI_Test).
 *   - Hardware support for asynchronous DMA or RDMA.
 *   - Whether the computation shares CPU/memory-bus resources with the
 *     communication, causing contention.
 *
 * For our dummy computation (a floating-point sum), on a typical shared-memory
 * system the observed wall-clock time for non-blocking communication may be
 * similar to blocking, because the MPI library may not progress the transfer
 * until MPI_Wait is called. On systems with true asynchronous progress (e.g.,
 * RDMA-capable networks, or MPI implementations with a progress thread), the
 * overlap benefit can be substantial.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define NUM_TRIALS 3
#define TAG 0
#define DUMMY_WORK_ITERS 1000000

/*
 * Dummy computation to perform while communication is in progress.
 * Returns a value to prevent the compiler from optimizing it away.
 */
static double dummy_computation(void)
{
    double sum = 0.0;
    for (int i = 0; i < DUMMY_WORK_ITERS; i++) {
        sum += sin((double)i) * cos((double)i);
    }
    return sum;
}

/*
 * Run one non-blocking send/recv trial. Returns elapsed wall-clock time.
 * dummy_result is set to the dummy computation result (to prevent optimization).
 */
static double run_trial(int *buf, int count, int rank, double *dummy_result)
{
    MPI_Request request;
    MPI_Status status;
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);  /* synchronize before timing */

    t_start = MPI_Wtime();

    if (rank == 0) {
        /* Initiate non-blocking send */
        MPI_Isend(buf, count, MPI_INT, 1, TAG, MPI_COMM_WORLD, &request);

        /* Perform computation while send is in progress */
        *dummy_result = dummy_computation();

        /* Wait for send to complete before reusing buffer */
        MPI_Wait(&request, &status);

    } else if (rank == 1) {
        /* Initiate non-blocking receive */
        MPI_Irecv(buf, count, MPI_INT, 0, TAG, MPI_COMM_WORLD, &request);

        /* Perform computation while receive is in progress */
        *dummy_result = dummy_computation();

        /* Wait for receive to complete before accessing data */
        MPI_Wait(&request, &status);
    }

    t_end = MPI_Wtime();

    return t_end - t_start;
}

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "Error: this program requires exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    /* Message sizes: 1 MB, 10 MB, 50 MB expressed as number of ints */
    int counts[] = {250000, 2500000, 12500000};
    const char *labels[] = {"1 MB", "10 MB", "50 MB"};
    int num_sizes = 3;

    for (int s = 0; s < num_sizes; s++) {
        int count = counts[s];
        int *buf = (int *)malloc(count * sizeof(int));
        if (!buf) {
            fprintf(stderr, "Rank %d: malloc failed for %s\n", rank, labels[s]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Process 0 initializes the array */
        if (rank == 0) {
            for (int i = 0; i < count; i++)
                buf[i] = i;
        }

        double total_time = 0.0;
        double dummy = 0.0;

        for (int t = 0; t < NUM_TRIALS; t++) {
            double elapsed = run_trial(buf, count, rank, &dummy);
            total_time += elapsed;
        }

        double avg_time = total_time / NUM_TRIALS;

        if (rank == 1) {
            printf("[Non-Blocking] Size: %s (%d ints)\n", labels[s], count);
            printf("  First element: %d, Last element: %d\n", buf[0], buf[count - 1]);
            printf("  Average transfer time over %d trials: %.6f seconds\n",
                   NUM_TRIALS, avg_time);
            printf("  (dummy computation result: %f — prevents optimization)\n\n", dummy);
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
