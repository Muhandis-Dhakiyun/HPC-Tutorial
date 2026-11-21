/*
 * Problem 5: Two-Dimensional Grid Topology
 *
 * Objective:
 *   Arrange processes in a 2D grid and exchange messages with valid neighbors
 *   in the up, down, left, and right directions.
 *
 * Tasks:
 *   13. Determine the row and column of each process.
 *   14. Identify valid neighbors and handle boundaries correctly (non-periodic).
 *   15. Exchange a message with each valid neighbor.
 *   16. Print each process rank and the ranks of its neighbors.
 *   17. Extension: investigate MPI_Cart_create and MPI_Cart_shift.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* Helper to convert neighbor rank to string for printing */
const char* rank_str(int r, char *buf) {
    if (r == MPI_PROC_NULL || r < 0) {
        return "NONE";
    }
    sprintf(buf, "%d", r);
    return buf;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Determine 2D grid dimensions (default: 2x2 for 4 processes) */
    int rows = 2, cols = 2;
    if (size == 4) {
        rows = 2; cols = 2;
    } else if (size == 6) {
        rows = 2; cols = 3;
    } else if (size == 8) {
        rows = 2; cols = 4;
    } else if (size == 9) {
        rows = 3; cols = 3;
    } else {
        /* Generic factorization */
        rows = 2;
        cols = size / 2;
        if (rows * cols != size) {
            if (rank == 0) {
                fprintf(stderr, "Error: Number of processes (%d) cannot form a simple 2D grid.\n", size);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("========================================================================\n");
        printf(" Problem 5: Two-Dimensional Grid Topology (%d x %d Grid, %d Processes)\n", rows, cols, size);
        printf("========================================================================\n");
        printf("Grid Layout:\n");
        for (int r = 0; r < rows; r++) {
            printf("  ");
            for (int c = 0; c < cols; c++) {
                printf("[Rank %d]\t", r * cols + c);
            }
            printf("\n");
        }
        printf("========================================================================\n\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* -------------------------------------------------------------------------
     * PART 1: Manual Coordinate Calculation and Neighbor Exchange
     * ------------------------------------------------------------------------- */

    /* Task 13: Determine row and column of each process */
    int my_row = rank / cols;
    int my_col = rank % cols;

    /* Task 14: Identify valid neighbors and handle boundaries (MPI_PROC_NULL) */
    int up    = (my_row > 0)        ? (my_row - 1) * cols + my_col : MPI_PROC_NULL;
    int down  = (my_row < rows - 1) ? (my_row + 1) * cols + my_col : MPI_PROC_NULL;
    int left  = (my_col > 0)        ? my_row * cols + (my_col - 1) : MPI_PROC_NULL;
    int right = (my_col < cols - 1) ? my_row * cols + (my_col + 1) : MPI_PROC_NULL;

    /* Task 15: Exchange a message with each valid neighbor */
    /*
     * We send our rank packed with a directional tag or data to each valid neighbor.
     * With MPI_PROC_NULL, MPI automatically treats non-existent neighbors as no-ops!
     */
    int send_val = rank * 10;
    int recv_up = -1, recv_down = -1, recv_left = -1, recv_right = -1;

    MPI_Request reqs[8];
    int req_count = 0;

    /* Non-blocking receives from valid neighbors */
    MPI_Irecv(&recv_up,    1, MPI_INT, up,    101, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Irecv(&recv_down,  1, MPI_INT, down,  102, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Irecv(&recv_left,  1, MPI_INT, left,  103, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Irecv(&recv_right, 1, MPI_INT, right, 104, MPI_COMM_WORLD, &reqs[req_count++]);

    /* Non-blocking sends to valid neighbors */
    MPI_Isend(&send_val, 1, MPI_INT, up,    102, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(&send_val, 1, MPI_INT, down,  101, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(&send_val, 1, MPI_INT, left,  104, MPI_COMM_WORLD, &reqs[req_count++]);
    MPI_Isend(&send_val, 1, MPI_INT, right, 103, MPI_COMM_WORLD, &reqs[req_count++]);

    MPI_Status statuses[8];
    /* Wait for all active communications to complete */
    MPI_Waitall(req_count, reqs, statuses);

    /* Task 16: Print each process rank, grid coordinates, and neighbor ranks */
    char b1[16], b2[16], b3[16], b4[16];
    for (int p = 0; p < size; p++) {
        if (rank == p) {
            printf("[Rank %d at (row %d, col %d)] Neighbors -> Up: %-4s Down: %-4s Left: %-4s Right: %-4s\n",
                   rank, my_row, my_col,
                   rank_str(up, b1), rank_str(down, b2), rank_str(left, b3), rank_str(right, b4));
            printf("         Received data -> from Up: %-4s from Down: %-4s from Left: %-4s from Right: %-4s\n",
                   (up != MPI_PROC_NULL ? rank_str(recv_up, b1) : "N/A"),
                   (down != MPI_PROC_NULL ? rank_str(recv_down, b2) : "N/A"),
                   (left != MPI_PROC_NULL ? rank_str(recv_left, b3) : "N/A"),
                   (right != MPI_PROC_NULL ? rank_str(recv_right, b4) : "N/A"));
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* -------------------------------------------------------------------------
     * PART 2 (Task 17 Extension): MPI Virtual Topologies (MPI_Cart_create & MPI_Cart_shift)
     * ------------------------------------------------------------------------- */
    if (rank == 0) {
        printf("\n------------------------------------------------------------------------\n");
        printf(" Task 17 Extension: MPI Cartesian Topology (MPI_Cart_create & MPI_Cart_shift)\n");
        printf("------------------------------------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int dims[2] = {rows, cols};
    int periods[2] = {0, 0}; /* Non-periodic boundaries */
    int reorder = 0;
    MPI_Comm cart_comm;

    /* Create Cartesian communicator */
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    int cart_coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, cart_coords);

    /* Find neighbors using MPI_Cart_shift */
    int cart_up, cart_down, cart_left, cart_right;
    /* Dimension 0 = vertical (rows): shift of +1 gives source=up, dest=down */
    MPI_Cart_shift(cart_comm, 0, 1, &cart_up, &cart_down);
    /* Dimension 1 = horizontal (cols): shift of +1 gives source=left, dest=right */
    MPI_Cart_shift(cart_comm, 1, 1, &cart_left, &cart_right);

    /* Exchange message along dimensions using MPI_Sendrecv */
    int c_recv_up = -1, c_recv_down = -1, c_recv_left = -1, c_recv_right = -1;
    MPI_Status st;

    /* Vertical exchange */
    MPI_Sendrecv(&send_val, 1, MPI_INT, cart_up, 201,
                 &c_recv_down, 1, MPI_INT, cart_down, 201, cart_comm, &st);
    MPI_Sendrecv(&send_val, 1, MPI_INT, cart_down, 202,
                 &c_recv_up, 1, MPI_INT, cart_up, 202, cart_comm, &st);

    /* Horizontal exchange */
    MPI_Sendrecv(&send_val, 1, MPI_INT, cart_left, 203,
                 &c_recv_right, 1, MPI_INT, cart_right, 203, cart_comm, &st);
    MPI_Sendrecv(&send_val, 1, MPI_INT, cart_right, 204,
                 &c_recv_left, 1, MPI_INT, cart_left, 204, cart_comm, &st);

    for (int p = 0; p < size; p++) {
        if (rank == p) {
            printf("[Cartesian Rank %d at (%d, %d)] Cart_shift -> Up: %-4s Down: %-4s Left: %-4s Right: %-4s\n",
                   rank, cart_coords[0], cart_coords[1],
                   rank_str(cart_up, b1), rank_str(cart_down, b2),
                   rank_str(cart_left, b3), rank_str(cart_right, b4));
            fflush(stdout);
        }
        MPI_Barrier(cart_comm);
    }

    if (rank == 0) {
        printf("========================================================================\n");
    }

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
