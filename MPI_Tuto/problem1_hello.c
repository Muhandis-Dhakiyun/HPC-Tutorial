/*
 * Problem 1: Hello World with MPI
 *
 * Objective:
 *   Learn the basic structure of an MPI program.
 *   Prints each process rank and the total number of processes.
 */

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;

    /* Initialize the MPI execution environment */
    MPI_Init(&argc, &argv);

    /* Determine the rank (unique identifier) of the calling process */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Determine the total number of processes in the communicator */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Print process rank and communicator size */
    printf("Hello from Process rank %d of %d processes!\n", rank, size);

    /* Terminate the MPI execution environment */
    MPI_Finalize();

    return 0;
}

