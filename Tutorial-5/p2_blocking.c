/*
 * Problem 2: Blocking Point-to-Point Communication
 *
 * Process 0 initializes an integer array and sends it to Process 1 using
 * MPI_Send. Process 1 receives with MPI_Recv, prints first/last elements,
 * and measures transfer time with MPI_Wtime.
 *
 * Tests three message sizes: 1 MB, 10 MB, 50 MB (as int arrays).
 * Runs 3 trials per size and reports average transfer time.
 *
 * Compile: mpicc -o p2_blocking p2_blocking.c
 * Run:     mpirun -np 2 ./p2_blocking
 *
 * ANALYSIS:
 *
 * 1) Why transfer time increases with message size:
 *    Larger messages require more data to be copied into internal MPI buffers,
 *    transmitted over the network (or shared-memory channel), and copied out
 *    at the receiver. The transfer time is dominated by the data volume
 *    divided by the available bandwidth, so more bytes = more time.
 *
 * 2) Does doubling the size double the time?
 *    Not exactly. Transfer time ≈ latency + (size / bandwidth). For small
 *    messages, the fixed latency dominates and doubling the size does NOT
 *    double the time. For very large messages, the bandwidth term dominates
 *    and time scales roughly linearly with size, so doubling the size
 *    approximately doubles the time. Protocol switches (eager vs rendezvous)
 *    and memory-copy overhead also introduce non-linearities.
 *
 * 3) Influence of latency, bandwidth, and memory copying:
 *    - Latency: the fixed startup cost per message (buffer allocation,
 *      handshake, system-call overhead). Important for small messages.
 *    - Bandwidth: the rate of sustained data transfer. Dominates for large
 *      messages. Physical limits (memory bus, network link) cap bandwidth.
 *    - Memory copying: MPI implementations often copy data into internal
 *      buffers (eager protocol) or use zero-copy (rendezvous protocol for
 *      large messages). Extra copies add overhead proportional to size.
 *
 * 4) What blocking means:
 *    MPI_Send is "blocking" in the sense that it does not return until the
 *    send buffer can be safely reused by the caller. This does NOT mean the
 *    matching MPI_Recv has completed. For small messages, MPI may buffer
 *    the data internally (eager protocol) so MPI_Send returns quickly. For
 *    large messages, MPI may use a rendezvous protocol where MPI_Send blocks
 *    until the receiver posts a matching MPI_Recv. MPI_Recv always blocks
 *    until the data has arrived and been placed in the user buffer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NUM_TRIALS 3
#define TAG 0

/*
 * Run one blocking send/recv trial. Returns elapsed wall-clock time.
 */
static double run_trial(int *buf, int count, int rank)
{
    double t_start, t_end;

    MPI_Barrier(MPI_COMM_WORLD);  /* synchronize before timing */

    t_start = MPI_Wtime();

    if (rank == 0) {
        MPI_Send(buf, count, MPI_INT, 1, TAG, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(buf, count, MPI_INT, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

        for (int t = 0; t < NUM_TRIALS; t++) {
            double elapsed = run_trial(buf, count, rank);
            total_time += elapsed;
        }

        double avg_time = total_time / NUM_TRIALS;

        if (rank == 1) {
            printf("[Blocking] Size: %s (%d ints)\n", labels[s], count);
            printf("  First element: %d, Last element: %d\n", buf[0], buf[count - 1]);
            printf("  Average transfer time over %d trials: %.6f seconds\n\n",
                   NUM_TRIALS, avg_time);
        }

        free(buf);
    }

    MPI_Finalize();
    return 0;
}
