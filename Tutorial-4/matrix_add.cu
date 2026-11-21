/*
 * Problem 2c: Matrix Addition
 * =============================
 * This program performs element-wise addition of two matrices A and B to
 * produce matrix C, where C[row][col] = A[row][col] + B[row][col].
 *
 * Each thread computes exactly one output element using 2D grid-block mapping.
 *
 * CUDA Memory Management Pattern:
 *   1. Allocate host arrays (malloc)
 *   2. Initialize host data
 *   3. Allocate device arrays (cudaMalloc)
 *   4. Copy input data host → device (cudaMemcpy H2D)
 *   5. Launch kernel on GPU
 *   6. Copy results device → host (cudaMemcpy D2H)
 *   7. Verify results on host
 *   8. Free device and host memory
 *
 * Compile: nvcc -o matrix_add matrix_add.cu
 * Run:     ./matrix_add
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1024           // Matrix dimension (N x N)
#define BLOCK_SIZE 16    // 16×16 threads per block

/*
 * Kernel: matrixAdd
 * Each thread computes one element of the output matrix C = A + B.
 *
 * Matrices are stored in row-major 1D arrays:
 *   element (row, col) is at index [row * width + col]
 *
 * Parameters:
 *   A, B  - input matrices (device pointers)
 *   C     - output matrix (device pointer)
 *   width - number of columns (= number of rows for square matrix)
 */
__global__ void matrixAdd(const float *A, const float *B, float *C, int width) {
    // Map thread to matrix element using 2D indexing
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: threads beyond the matrix dimensions must not access memory
    if (row < width && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(float);

    printf("=== Matrix Addition (C = A + B) ===\n");
    printf("Matrix size: %d x %d\n", N, N);
    printf("Block size:  %d x %d threads\n", BLOCK_SIZE, BLOCK_SIZE);

    // ---- Step 1: Allocate host memory ----
    float *h_A = (float *)malloc(matrixBytes);
    float *h_B = (float *)malloc(matrixBytes);
    float *h_C = (float *)malloc(matrixBytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return 1;
    }

    // ---- Step 2: Initialize host matrices ----
    for (int i = 0; i < matrixSize; i++) {
        h_A[i] = (float)(i % 100);           // A values: 0, 1, ..., 99, 0, 1, ...
        h_B[i] = (float)((i * 2) % 100);     // B values: 0, 2, 4, ..., 98, 0, 2, ...
    }

    // ---- Step 3: Allocate device memory ----
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixBytes);
    cudaMalloc((void **)&d_B, matrixBytes);
    cudaMalloc((void **)&d_C, matrixBytes);

    // ---- Step 4: Copy input matrices from host to device ----
    cudaMemcpy(d_A, h_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixBytes, cudaMemcpyHostToDevice);

    // ---- Step 5: Configure and launch kernel ----
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,   // ceil(N / BLOCK_SIZE)
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    printf("Grid size:   %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Total threads: %d\n\n", gridDim.x * blockDim.x * gridDim.y * blockDim.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrixAdd<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    // Wait for kernel to complete
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.4f ms\n\n", milliseconds);

    // ---- Step 6: Copy result matrix from device to host ----
    cudaMemcpy(h_C, d_C, matrixBytes, cudaMemcpyDeviceToHost);

    // ---- Step 7: Verify results on host ----
    printf("Verifying results on host...\n");
    int errors = 0;
    for (int i = 0; i < matrixSize; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            if (errors < 10) {  // Print first 10 mismatches
                int row = i / N;
                int col = i % N;
                printf("  MISMATCH at (%d,%d): C=%.2f, expected=%.2f\n",
                       row, col, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("VERIFICATION PASSED: All %d elements are correct.\n", matrixSize);
    } else {
        printf("VERIFICATION FAILED: %d elements have incorrect values.\n", errors);
    }

    // Print a few sample results
    printf("\nSample results (row, col): A + B = C\n");
    printf("%-15s %-10s %-10s %-10s\n", "(row, col)", "A", "B", "C");
    printf("----------------------------------------------\n");
    int samples[][2] = {{0,0}, {0,1}, {10,10}, {100,100}, {N-1,N-1}};
    for (int s = 0; s < 5; s++) {
        int r = samples[s][0], c = samples[s][1];
        int idx = r * N + c;
        printf("(%4d, %4d)    %-10.2f %-10.2f %-10.2f\n",
               r, c, h_A[idx], h_B[idx], h_C[idx]);
    }

    // ---- Step 8: Cleanup ----
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nDone.\n");
    return 0;
}
