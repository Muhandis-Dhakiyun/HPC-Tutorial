/*
 * Problem 2a: 2D Global Index Mapping
 * =====================================
 * This program demonstrates 2D thread-block indexing in CUDA.
 *
 * In many GPU applications (images, matrices), data is 2D. CUDA supports
 * multi-dimensional grids and blocks so each thread can naturally map to
 * a 2D element.
 *
 * 2D Index Calculation:
 *   col = blockIdx.x * blockDim.x + threadIdx.x  (column / x-dimension)
 *   row = blockIdx.y * blockDim.y + threadIdx.y  (row    / y-dimension)
 *
 * Example with 8×8 matrix, 4×4 thread blocks → 2×2 grid of blocks:
 *
 *   Grid (2×2 blocks):          Each Block (4×4 threads):
 *   +--------+--------+        +---------+---------+---------+---------+
 *   | (0,0)  | (1,0)  |        | (0,0)   | (1,0)   | (2,0)   | (3,0)   |
 *   +--------+--------+        +---------+---------+---------+---------+
 *   | (0,1)  | (1,1)  |        | (0,1)   | (1,1)   | (2,1)   | (3,1)   |
 *   +--------+--------+        +---------+---------+---------+---------+
 *                               | (0,2)   | (1,2)   | (2,2)   | (3,2)   |
 *                               +---------+---------+---------+---------+
 *                               | (0,3)   | (1,3)   | (2,3)   | (3,3)   |
 *                               +---------+---------+---------+---------+
 *
 * Compile: nvcc -o global_index_2d global_index_2d.cu
 * Run:     ./global_index_2d
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Matrix dimensions
#define MATRIX_WIDTH  8
#define MATRIX_HEIGHT 8

// Block dimensions (threads per block in each dimension)
#define BLOCK_SIZE_X 4
#define BLOCK_SIZE_Y 4

/*
 * Kernel: print2DIndex
 * Each thread computes its global (row, col) position and prints it.
 * Threads outside the matrix bounds are skipped (boundary check).
 */
__global__ void print2DIndex(int width, int height) {
    // Compute global column and row indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: only threads mapping to valid matrix elements should act
    if (col < width && row < height) {
        printf("Block(%d,%d) Thread(%d,%d) -> Global Position (row=%d, col=%d) "
               "-> Linear index = %d\n",
               blockIdx.x, blockIdx.y,
               threadIdx.x, threadIdx.y,
               row, col,
               row * width + col);  // row-major linear index
    }
}

int main() {
    printf("=== 2D Global Index Mapping ===\n");
    printf("Matrix size: %d x %d\n", MATRIX_HEIGHT, MATRIX_WIDTH);
    printf("Block size:  %d x %d threads\n", BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // Calculate grid dimensions:
    // We need enough blocks to cover the entire matrix
    // Grid.x = ceil(width  / blockDim.x)
    // Grid.y = ceil(height / blockDim.y)
    dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 gridDim(
        (MATRIX_WIDTH  + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,  // = 2
        (MATRIX_HEIGHT + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y   // = 2
    );

    printf("Grid size:   %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Total threads: %d\n\n", gridDim.x * blockDim.x * gridDim.y * blockDim.y);

    // Launch the 2D kernel
    // <<<gridDim, blockDim>>> uses dim3 for multi-dimensional configuration
    print2DIndex<<<gridDim, blockDim>>>(MATRIX_WIDTH, MATRIX_HEIGHT);

    // Wait for GPU to finish and flush printf output
    cudaDeviceSynchronize();

    printf("\nAll %d matrix elements have been mapped to unique threads.\n",
           MATRIX_WIDTH * MATRIX_HEIGHT);

    return 0;
}
