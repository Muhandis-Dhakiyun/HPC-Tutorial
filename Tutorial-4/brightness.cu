/*
 * Problem 2b: Image Brightness Increase
 * =======================================
 * This program simulates increasing the brightness of a grayscale image
 * using CUDA. Each pixel value is an unsigned char (0–255).
 *
 * Approach:
 *   - Image is 512×512 pixels (grayscale, 1 byte per pixel)
 *   - Each thread processes exactly one pixel
 *   - Thread block: 16×16 = 256 threads
 *   - Grid: 32×32 blocks → 32*16 = 512 threads per dimension → covers 512×512
 *   - Brightness is increased by a fixed offset (e.g., +50)
 *   - Result is clamped to 255 to prevent overflow
 *
 * Memory management:
 *   1. Allocate host memory and initialize pixel data
 *   2. Allocate device memory with cudaMalloc
 *   3. Copy host → device with cudaMemcpy
 *   4. Launch kernel
 *   5. Copy device → host with cudaMemcpy
 *   6. Print sample pixels and free memory
 *
 * Compile: nvcc -o brightness brightness.cu
 * Run:     ./brightness
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define IMG_WIDTH   512
#define IMG_HEIGHT  512
#define BLOCK_SIZE  16        // 16×16 threads per block
#define BRIGHTNESS  50        // brightness increase value

/*
 * Kernel: adjustBrightness
 * Each thread processes one pixel: adds the brightness offset and clamps to 255.
 *
 * Parameters:
 *   image  - pointer to the image pixel array in device memory
 *   width  - image width in pixels
 *   height - image height in pixels
 *   offset - brightness value to add (can be negative for darkening)
 */
__global__ void adjustBrightness(unsigned char *image, int width, int height, int offset) {
    // 2D thread indexing to map each thread to a pixel
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (col < width && row < height) {
        // Convert 2D (row, col) to 1D linear index (row-major order)
        int idx = row * width + col;

        // Read original pixel value
        int pixel = image[idx];

        // Add brightness offset
        pixel += offset;

        // Clamp to valid range [0, 255]
        // Using min/max is cleaner than if-else
        if (pixel > 255) pixel = 255;
        if (pixel < 0)   pixel = 0;

        // Write back the adjusted pixel
        image[idx] = (unsigned char)pixel;
    }
}

int main() {
    // ---- Host memory allocation ----
    int imageSize = IMG_WIDTH * IMG_HEIGHT;
    size_t imageBytes = imageSize * sizeof(unsigned char);

    unsigned char *h_image   = (unsigned char *)malloc(imageBytes);  // working copy
    unsigned char *h_original = (unsigned char *)malloc(imageBytes); // keep original for comparison

    // Initialize image with sample pixel values (gradient pattern)
    for (int i = 0; i < imageSize; i++) {
        h_image[i] = (unsigned char)(i % 256);  // values cycle 0–255
    }

    // Save original for comparison
    memcpy(h_original, h_image, imageBytes);

    printf("=== Image Brightness Increase ===\n");
    printf("Image size: %d x %d pixels\n", IMG_WIDTH, IMG_HEIGHT);
    printf("Brightness offset: +%d\n", BRIGHTNESS);
    printf("Block size: %d x %d threads\n", BLOCK_SIZE, BLOCK_SIZE);

    // ---- Device memory allocation ----
    unsigned char *d_image;
    cudaMalloc((void **)&d_image, imageBytes);

    // ---- Copy image from host to device ----
    cudaMemcpy(d_image, h_image, imageBytes, cudaMemcpyHostToDevice);

    // ---- Kernel launch configuration ----
    // 16×16 threads per block, 32×32 blocks in the grid
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);              // (16, 16)
    dim3 gridDim(IMG_WIDTH / BLOCK_SIZE, IMG_HEIGHT / BLOCK_SIZE);  // (32, 32)

    printf("Grid size: %d x %d blocks\n", gridDim.x, gridDim.y);
    printf("Total threads: %d\n\n", gridDim.x * blockDim.x * gridDim.y * blockDim.y);

    // Launch kernel
    adjustBrightness<<<gridDim, blockDim>>>(d_image, IMG_WIDTH, IMG_HEIGHT, BRIGHTNESS);

    // ---- Copy result from device to host ----
    cudaMemcpy(h_image, d_image, imageBytes, cudaMemcpyDeviceToHost);

    // ---- Print before/after for sample pixels ----
    printf("Sample pixel values (before -> after brightness adjustment):\n");
    printf("%-20s %-10s %-10s %-10s\n", "Position (row,col)", "Before", "After", "Expected");
    printf("------------------------------------------------------------\n");

    // Pick a few representative pixels to display
    int sampleRows[] = {0, 0, 100, 255, 400, 511, 511};
    int sampleCols[] = {0, 100, 200, 255, 300, 400, 511};
    int numSamples = 7;

    int errors = 0;
    for (int s = 0; s < numSamples; s++) {
        int r = sampleRows[s];
        int c = sampleCols[s];
        int idx = r * IMG_WIDTH + c;

        int original = h_original[idx];
        int adjusted = h_image[idx];
        int expected = original + BRIGHTNESS;
        if (expected > 255) expected = 255;

        printf("(%3d, %3d)           %-10d %-10d %-10d %s\n",
               r, c, original, adjusted, expected,
               (adjusted == expected) ? "OK" : "MISMATCH");

        if (adjusted != expected) errors++;
    }

    printf("\n");

    // ---- Full verification ----
    for (int i = 0; i < imageSize; i++) {
        int expected = h_original[i] + BRIGHTNESS;
        if (expected > 255) expected = 255;
        if (h_image[i] != (unsigned char)expected) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("VERIFICATION PASSED: All %d pixels correctly adjusted.\n", imageSize);
    } else {
        printf("VERIFICATION FAILED: %d pixels have incorrect values.\n", errors);
    }

    // ---- Cleanup ----
    cudaFree(d_image);
    free(h_image);
    free(h_original);

    return 0;
}
