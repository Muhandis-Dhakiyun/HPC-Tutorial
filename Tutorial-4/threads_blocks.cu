/*
 * Problem 1: Exploring Threads and Blocks
 * =========================================
 * This program demonstrates how CUDA threads identify themselves using
 * built-in variables:
 *   - threadIdx.x : thread index within the block (0 to blockDim.x - 1)
 *   - blockIdx.x  : block index within the grid  (0 to gridDim.x - 1)
 *   - blockDim.x  : number of threads per block
 *   - globalIndex  : unique thread ID = blockIdx.x * blockDim.x + threadIdx.x
 *
 * CUDA Execution Model:
 *   A kernel is launched with a grid of blocks, each block containing threads.
 *   Grid:  [Block 0] [Block 1] ... [Block N-1]
 *   Block: [Thread 0] [Thread 1] ... [Thread M-1]
 *
 *   The global index uniquely identifies each thread across the entire grid.
 *
 * We launch with 2 blocks of 4 threads each (8 threads total) so the
 * output is small and readable.
 *
 * Compile: nvcc -o threads_blocks threads_blocks.cu
 * Run:     ./threads_blocks
 */

#include <stdio.h>
#include <cuda_runtime.h>

/*
 * Kernel: printThreadInfo
 * Each thread prints its identification variables.
 * Note: printf from device code requires compute capability >= 2.0.
 * Output order is non-deterministic because threads run in parallel.
 */
__global__ void printThreadInfo() {
    // Calculate the global (unique) index of this thread
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Thread: blockIdx.x = %d, threadIdx.x = %d, blockDim.x = %d, "
           "globalIndex = %d\n",
           blockIdx.x, threadIdx.x, blockDim.x, globalIdx);
}

int main() {
    // ---- Kernel launch configuration ----
    // We use a 1D grid with 2 blocks, each containing 4 threads
    int numBlocks = 2;
    int threadsPerBlock = 4;
    int totalThreads = numBlocks * threadsPerBlock;

    printf("=== Exploring Threads and Blocks ===\n");
    printf("Launching kernel with %d blocks of %d threads each (%d threads total)\n\n",
           numBlocks, threadsPerBlock, totalThreads);

    printf("Expected global indices: 0 through %d\n", totalThreads - 1);
    printf("Block 0: threads 0-3  (globalIndex 0-3)\n");
    printf("Block 1: threads 0-3  (globalIndex 4-7)\n\n");

    // Launch the kernel
    // Syntax: kernel<<<numBlocks, threadsPerBlock>>>()
    printThreadInfo<<<numBlocks, threadsPerBlock>>>();

    // cudaDeviceSynchronize() blocks the host until all device work completes.
    // This is necessary to ensure all printf output from the GPU is flushed
    // before the program exits.
    cudaDeviceSynchronize();

    printf("\nNote: Thread output order may vary because GPU threads execute in parallel.\n");

    return 0;
}
