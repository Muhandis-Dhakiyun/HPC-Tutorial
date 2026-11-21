/*
 * Problem 0: Device Query
 * ========================
 * This program queries all CUDA-capable devices on the system and displays
 * key hardware properties. It uses the CUDA Runtime API functions:
 *   - cudaGetDeviceCount(): returns the number of CUDA-capable GPUs
 *   - cudaGetDeviceProperties(): fills a cudaDeviceProp struct with hardware info
 *
 * Compile: nvcc -o device_query device_query.cu
 * Run:     ./device_query
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;

    // Query how many CUDA-capable devices are available
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 1;
    }

    printf("=== CUDA Device Query ===\n");
    printf("Number of CUDA-capable devices: %d\n\n", deviceCount);

    // Iterate over each device and print its properties
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("---------- Device %d ----------\n", dev);

        // Device name (e.g., "NVIDIA GeForce RTX 3080")
        printf("  Device Name:                 %s\n", prop.name);

        // Compute capability — major.minor version that determines available features
        printf("  Compute Capability:          %d.%d\n", prop.major, prop.minor);

        // Total global memory — the main GPU memory (VRAM)
        printf("  Total Global Memory:         %.2f MB (%zu bytes)\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0),
               prop.totalGlobalMem);

        // Shared memory per block — fast on-chip memory shared by threads in a block
        printf("  Shared Memory per Block:     %zu bytes\n", prop.sharedMemPerBlock);

        // Number of streaming multiprocessors (SMs) — the main compute units on the GPU
        printf("  Number of Multiprocessors:   %d\n", prop.multiProcessorCount);

        // Maximum threads per block — hardware limit on block size
        printf("  Max Threads per Block:       %d\n", prop.maxThreadsPerBlock);

        // Max grid size — the maximum number of blocks in each dimension of the grid
        printf("  Max Grid Size:               (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // Max block dimensions — the maximum number of threads in each dimension of a block
        printf("  Max Block Dimensions:        (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

        // Warp size — the number of threads that execute in lockstep (SIMT width)
        // On all current NVIDIA GPUs this is 32
        printf("  Warp Size:                   %d\n", prop.warpSize);

        // Additional useful properties
        printf("  Clock Rate:                  %.2f GHz\n", prop.clockRate / 1e6);
        printf("  Memory Clock Rate:           %.2f GHz\n", prop.memoryClockRate / 1e6);
        printf("  Memory Bus Width:            %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size:               %d bytes\n", prop.l2CacheSize);

        printf("\n");
    }

    return 0;
}
