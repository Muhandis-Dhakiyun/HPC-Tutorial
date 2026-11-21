/*
 * Problem 4: Ring Communication Topology
 *
 * Objective:
 *   Connect processes in a logical ring. Each process sends a message to its
 *   next neighbor and receives a message from its previous neighbor.
 *
 * Topology mapping:
 *   next     = (rank + 1) % size;
 *   previous = (rank - 1 + size) % size;
 *
 * Deadlock prevention:
 *   This program demonstrates two safe approaches:
 *     1. MPI_Sendrecv: standard MPI function designed specifically to avoid deadlock.
 *     2. Alternating even/odd ordering: breaks the circular wait condition.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) {
            fprintf(stderr, "Note: Tutorial specifies testing with at least 4 processes. Current size = %d.\n", size);
        }
        if (size < 2) {
            MPI_Finalize();
            return 1;
        }
    }

    /* Compute logical ring neighbors */
    int next = (rank + 1) % size;
    int previous = (rank - 1 + size) % size;

    if (rank == 0) {
        printf("========================================================================\n");
        printf(" Problem 4: Ring Communication Topology (Total Processes: %d)\n", size);
        printf("========================================================================\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------
     * METHOD 1: Safe Ring Exchange using MPI_Sendrecv
     * ------------------------------------------------------------------------- */
    int send_val = (rank + 1) * 10; /* Each process sends its unique data */
    int recv_val = -1;
    MPI_Status status;

    /*
     * MPI_Sendrecv guarantees that send and receive are initiated without
     * creating a circular blocking dependency.
     */
    MPI_Sendrecv(&send_val, 1, MPI_INT, next, 0,
                 &recv_val, 1, MPI_INT, previous, 0,
                 MPI_COMM_WORLD, &status);

    /* Synchronize prints across ranks */
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            printf("[MPI_Sendrecv] Process %d (prev: %d, next: %d) sent %d to Process %d and received %d from Process %d\n",
                   rank, previous, next, send_val, next, recv_val, status.MPI_SOURCE);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("------------------------------------------------------------------------\n");
        printf(" METHOD 2: Ring Exchange using Even-Odd Phase Ordering\n");
        printf("------------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------
     * METHOD 2: Safe Ring Exchange using Even/Odd Alternating Ordering
     *
     * In a ring, if all processes call MPI_Send first, circular wait (deadlock) occurs.
     * By having even ranks Send then Recv, and odd ranks Recv then Send,
     * the cyclic dependency is broken for any pair of neighbors.
     * ------------------------------------------------------------------------- */
    int send_val2 = (rank * 100) + 7;
    int recv_val2 = -1;

    if (rank % 2 == 0) {
        /* Even ranks: SEND to next, then RECV from previous */
        MPI_Send(&send_val2, 1, MPI_INT, next, 10, MPI_COMM_WORLD);
        MPI_Recv(&recv_val2, 1, MPI_INT, previous, 10, MPI_COMM_WORLD, &status);
    } else {
        /* Odd ranks: RECV from previous, then SEND to next */
        MPI_Recv(&recv_val2, 1, MPI_INT, previous, 10, MPI_COMM_WORLD, &status);
        MPI_Send(&send_val2, 1, MPI_INT, next, 10, MPI_COMM_WORLD);
    }

    /* Print verification */
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            printf("[Even/Odd Order] Process %d sent %d to Process %d and received %d from Process %d\n",
                   rank, send_val2, next, recv_val2, status.MPI_SOURCE);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* -------------------------------------------------------------------------
     * BONUS: Multi-hop Token Pass around the full ring
     * The token travels size hops so every process visits and increments it.
     * ------------------------------------------------------------------------- */
    if (rank == 0) {
        printf("------------------------------------------------------------------------\n");
        printf(" BONUS: Token Circulation around Entire Ring (%d Hops)\n", size);
        printf("------------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int token;
    if (rank == 0) {
        token = 1000;
        printf("Process 0 initiating token = %d -> sending to Process %d\n", token, next);
        MPI_Send(&token, 1, MPI_INT, next, 99, MPI_COMM_WORLD);
        MPI_Recv(&token, 1, MPI_INT, previous, 99, MPI_COMM_WORLD, &status);
        printf("Process 0 received completed token = %d back from Process %d!\n", token, previous);
    } else {
        MPI_Recv(&token, 1, MPI_INT, previous, 99, MPI_COMM_WORLD, &status);
        token += 1;
        printf("Process %d received token = %d from Process %d -> forwarding %d to Process %d\n",
               rank, token - 1, previous, token, next);
        MPI_Send(&token, 1, MPI_INT, next, 99, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("========================================================================\n");
    }

    MPI_Finalize();
    return 0;
}

