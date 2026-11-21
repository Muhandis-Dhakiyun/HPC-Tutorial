/*
 * Problem 4: Ring Communication
 *
 * Processes are connected in a logical ring. Each process sends its rank
 * to the next process (rank+1)%size and receives from the previous process
 * (rank-1+size)%size. MPI_Sendrecv is used to avoid deadlock.
 *
 * Compile: mpicc -o p4_ring p4_ring.c
 * Run:     mpirun -np 4 ./p4_ring
 *
 * NOTES:
 * - MPI_Sendrecv performs a send and receive in a single call, allowing
 *   the MPI library to manage the ordering and avoid deadlocks that
 *   would occur if all processes called MPI_Send before MPI_Recv (since
 *   MPI_Send may block for large messages under rendezvous protocol).
 * - The ring topology is a common pattern in parallel computing, used in
 *   algorithms like token passing, pipeline processing, and parallel prefix.
 */

#include <stdio.h>
#include <mpi.h>

#define TAG 0

int main(int argc, char *argv[])
{
    int rank, size;
    int send_val, recv_val;
    int next, prev;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine neighbors in the ring */
    next = (rank + 1) % size;
    prev = (rank - 1 + size) % size;

    /* Each process sends its own rank */
    send_val = rank;

    /*
     * MPI_Sendrecv: simultaneously sends send_val to 'next' and receives
     * into recv_val from 'prev'. This avoids the deadlock that would occur
     * if every process called MPI_Send (which may block) before MPI_Recv.
     */
    MPI_Sendrecv(&send_val, 1, MPI_INT, next, TAG,     /* send */
                 &recv_val, 1, MPI_INT, prev, TAG,     /* recv */
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d: sent %d to process %d, received %d from process %d\n",
           rank, send_val, next, recv_val, prev);

    MPI_Finalize();
    return 0;
}
