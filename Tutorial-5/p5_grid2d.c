/*
 * Problem 5: 2D Grid Topology
 *
 * Arranges processes in a 2D Cartesian grid using MPI_Cart_create.
 * Each process identifies its neighbors (up, down, left, right) using
 * MPI_Cart_shift and exchanges messages with them using MPI_Sendrecv.
 * Boundaries are handled with MPI_PROC_NULL (sends/recvs to/from
 * MPI_PROC_NULL are no-ops).
 *
 * Compile: mpicc -o p5_grid2d p5_grid2d.c
 * Run:     mpirun -np 4 ./p5_grid2d      (creates a 2×2 grid)
 *          mpirun -np 9 ./p5_grid2d      (creates a 3×3 grid)
 *
 * NOTES:
 * - MPI_Cart_create creates a new communicator with Cartesian topology.
 *   The 'periods' array determines if the grid wraps around (periodic
 *   boundary). Here we use non-periodic (periods = {0, 0}).
 * - MPI_Cart_shift returns the source and destination ranks for a shift
 *   operation along a given dimension. For a non-periodic grid, processes
 *   at the edge get MPI_PROC_NULL as their missing neighbor.
 * - MPI_Cart_coords converts a rank to its (row, col) coordinates.
 * - MPI_Sendrecv is used for neighbor exchange to prevent deadlock.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TAG 0

int main(int argc, char *argv[])
{
    int rank, size;
    int dims[2] = {0, 0};       /* let MPI choose the grid dimensions */
    int periods[2] = {0, 0};    /* non-periodic (no wraparound) */
    int coords[2];              /* this process's (row, col) in the grid */
    int reorder = 1;            /* allow MPI to reorder ranks for efficiency */

    MPI_Comm cart_comm;         /* Cartesian communicator */
    int cart_rank;              /* rank in the Cartesian communicator */

    /* Neighbor ranks */
    int up, down, left, right;

    /* Values to send/receive */
    int send_val, recv_val;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
     * Let MPI determine balanced grid dimensions.
     * For 4 processes, MPI_Dims_create typically gives 2×2.
     * For 9 processes, it gives 3×3.
     */
    MPI_Dims_create(size, 2, dims);

    if (rank == 0) {
        printf("Creating a %d x %d process grid (%d processes total)\n\n",
               dims[0], dims[1], size);
    }

    /* Create the Cartesian topology communicator */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    /* Get this process's rank and coordinates in the Cartesian communicator */
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);

    /*
     * MPI_Cart_shift(comm, direction, displacement, &source, &dest)
     *   direction 0 = row (vertical: up/down)
     *   direction 1 = column (horizontal: left/right)
     *   displacement = 1 means shift by 1 step
     *
     * For non-periodic grids, MPI_PROC_NULL is returned for missing neighbors.
     */
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);     /* vertical neighbors */
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);   /* horizontal neighbors */

    printf("Rank %d (cart_rank %d) at grid position (%d, %d): "
           "up=%d, down=%d, left=%d, right=%d\n",
           rank, cart_rank, coords[0], coords[1],
           up, down, left, right);

    /*
     * Exchange data with each neighbor using MPI_Sendrecv.
     * Each process sends its cart_rank to each neighbor and receives
     * the neighbor's cart_rank. MPI_PROC_NULL causes the operation
     * to be a no-op for that direction.
     */
    send_val = cart_rank;

    /* Exchange with UP neighbor */
    MPI_Sendrecv(&send_val, 1, MPI_INT, up, TAG,
                 &recv_val, 1, MPI_INT, up, TAG,
                 cart_comm, MPI_STATUS_IGNORE);
    if (up != MPI_PROC_NULL)
        printf("  Rank %d: exchanged with UP neighbor %d, received %d\n",
               cart_rank, up, recv_val);

    /* Exchange with DOWN neighbor */
    MPI_Sendrecv(&send_val, 1, MPI_INT, down, TAG,
                 &recv_val, 1, MPI_INT, down, TAG,
                 cart_comm, MPI_STATUS_IGNORE);
    if (down != MPI_PROC_NULL)
        printf("  Rank %d: exchanged with DOWN neighbor %d, received %d\n",
               cart_rank, down, recv_val);

    /* Exchange with LEFT neighbor */
    MPI_Sendrecv(&send_val, 1, MPI_INT, left, TAG,
                 &recv_val, 1, MPI_INT, left, TAG,
                 cart_comm, MPI_STATUS_IGNORE);
    if (left != MPI_PROC_NULL)
        printf("  Rank %d: exchanged with LEFT neighbor %d, received %d\n",
               cart_rank, left, recv_val);

    /* Exchange with RIGHT neighbor */
    MPI_Sendrecv(&send_val, 1, MPI_INT, right, TAG,
                 &recv_val, 1, MPI_INT, right, TAG,
                 cart_comm, MPI_STATUS_IGNORE);
    if (right != MPI_PROC_NULL)
        printf("  Rank %d: exchanged with RIGHT neighbor %d, received %d\n",
               cart_rank, right, recv_val);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
