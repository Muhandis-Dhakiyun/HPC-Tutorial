/*
 * Problem 1: MPI Hello World
 *
 * Basic MPI program where each process prints its rank and total number
 * of processes. Demonstrates MPI_Init, MPI_Comm_rank, MPI_Comm_size,
 * MPI_Finalize.
 *
 * Compile: mpicc -o p1_hello p1_hello.c
 * Run:     mpirun -np 4 ./p1_hello
 *
 * WHY DOES THE SAME PROGRAM PRINT DIFFERENT RANKS?
 * -------------------------------------------------
 * When you launch an MPI program with mpirun (or mpiexec), the MPI runtime
 * spawns multiple copies (processes) of the same executable. Each process
 * is an independent OS process with its own address space. The MPI runtime
 * assigns each process a unique integer identifier called its "rank" within
 * the communicator (MPI_COMM_WORLD). Although every process runs the same
 * compiled binary, calling MPI_Comm_rank returns a different value in each
 * process, allowing them to branch into different code paths (SPMD model:
 * Single Program, Multiple Data). This is why the same printf statement
 * produces different output from each process.
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;

    /* Initialize the MPI execution environment */
    MPI_Init(&argc, &argv);

    /* Get the rank (ID) of this process within MPI_COMM_WORLD */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Get the total number of processes in MPI_COMM_WORLD */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from process %d of %d\n", rank, size);

    /* Clean up MPI environment */
    MPI_Finalize();

    return 0;
}
