# HPC Tutorial 4: GPU Programming with NVIDIA CUDA — Exploring Threads and Blocks

## High Performance Computing Laboratory Report & Detailed Solutions

---

## 1. Executive Summary & Overview

NVIDIA CUDA (Compute Unified Device Architecture) exposes the massive parallelism of modern Graphics Processing Units (GPUs) through a SIMT (Single Instruction, Multiple Threads) execution model. This laboratory covers:
1. **GPU Architecture & Device Querying**: Interrogating device compute capabilities, streaming multiprocessors (SMs), shared memory, and hardware warp limits.
2. **Thread Identification & 1D Indexing**: Exploring built-in CUDA runtime variables (`threadIdx`, `blockIdx`, `blockDim`, `gridDim`).
3. **2D Grid-Block Geometry**: Mapping 2D matrix coordinates to thread blocks for spatial data structures.
4. **Image Brightness Transformation**: Parallel 2D pixel processing on a $512 \times 512$ image ($262,144$ threads).
5. **Element-wise Matrix Addition**: Parallel kernel computing $C = A + B$ on $1024 \times 1024$ float matrices ($1,048,576$ elements).

---

## 2. Directory Structure

```text
Tutorial-4/
├── plots/
│   ├── plot1_cuda_thread_hierarchy.png       # CUDA Grid -> Blocks -> Threads hierarchy
│   ├── plot2_cuda_2d_mapping.png             # 2D Grid-to-Matrix coordinate mapping diagram
│   ├── plot3_image_brightness_histogram.png  # Image pixel distribution before and after brightness
│   └── plot4_matrix_add_verification.png     # Heatmaps verifying C = A + B matrix addition
├── device_query.cu                           # Program querying GPU hardware architecture
├── threads_blocks.cu                         # Problem 1: 1D thread identification kernel
├── global_index_2d.cu                        # Problem 2a: 2D global coordinate mapping
├── brightness.cu                             # Problem 2b: 512x512 image brightness transformation
├── matrix_add.cu                             # Problem 2c: 1024x1024 element-wise matrix addition
├── Cuda_ Exploring Threads and Blocks_.docx  # Original laboratory specification
├── Results.pdf                               # Formal publication-quality laboratory report
└── README.md                                 # Full documentation & verification results
```

---

## 3. Hardware Architecture & Device Query

Querying CUDA-capable devices via `cudaGetDeviceProperties` yields critical architectural specifications:

| Hardware Property | Enterprise GPU (Tesla A100 / RTX 4090) | Mid-Range GPU (Tesla T4 / RTX 3060) | Significance in CUDA Programming |
|:---|:---:|:---:|:---|
| **Compute Capability** | 8.0 / 8.9 | 7.5 / 8.6 | Determines supported PTX ISA instructions & features |
| **Total Global Memory** | 40 GB / 24 GB HBM2e/GDDR6X | 16 GB / 12 GB GDDR6 | Capacity of device heap accessible by all threads |
| **Streaming Multiprocessors (SMs)** | 108 / 128 | 40 / 28 | Hardware execution units managing active warps |
| **Shared Memory per Block** | 48 KB – 164 KB | 48 KB – 64 KB | High-speed on-chip cache shared within a block |
| **Max Threads per Block** | 1024 | 1024 | Hardware boundary for block dimension product ($X \times Y \times Z$) |
| **Warp Size** | 32 | 32 | Fundamental SIMD execution unit lockstep width |
| **Max Grid Dimensions** | $(2^{31}-1, 65535, 65535)$ | $(2^{31}-1, 65535, 65535)$ | Maximum allowable thread blocks per kernel launch |

---

## 4. Problem 1: Exploring 1D Threads and Blocks

### 4.1 Indexing Formula
When launching a 1D grid of blocks where each block has `blockDim.x` threads:
$$\text{globalIndex} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$

### 4.2 Kernel Launch & Output (2 Blocks of 4 Threads)
```c
printThreadInfo<<<2, 4>>>();
```
Execution trace:
```text
Thread: blockIdx.x = 0, threadIdx.x = 0, blockDim.x = 4, globalIndex = 0
Thread: blockIdx.x = 0, threadIdx.x = 1, blockDim.x = 4, globalIndex = 1
Thread: blockIdx.x = 0, threadIdx.x = 2, blockDim.x = 4, globalIndex = 2
Thread: blockIdx.x = 0, threadIdx.x = 3, blockDim.x = 4, globalIndex = 3
Thread: blockIdx.x = 1, threadIdx.x = 0, blockDim.x = 4, globalIndex = 4
Thread: blockIdx.x = 1, threadIdx.x = 1, blockDim.x = 4, globalIndex = 5
Thread: blockIdx.x = 1, threadIdx.x = 2, blockDim.x = 4, globalIndex = 6
Thread: blockIdx.x = 1, threadIdx.x = 3, blockDim.x = 4, globalIndex = 7
```

---

## 5. Problem 2: 2D Grid-Block Geometry and Applications

### 5.1 2D Global Index Derivation
For 2D data (images, matrices), CUDA utilizes `dim3` structs:
```c
dim3 block(16, 16);    // 256 threads per block
dim3 grid(32, 32);     // 1024 blocks -> 262,144 total threads
```
The unique 2D spatial coordinate $(r, c)$ and linear row-major memory offset are:
$$\text{col} = \text{blockIdx.x} \times \text{blockDim.x} + \text{threadIdx.x}$$
$$\text{row} = \text{blockIdx.y} \times \text{blockDim.y} + \text{threadIdx.y}$$
$$\text{linear\_idx} = \text{row} \times \text{width} + \text{col}$$

### 5.2 Problem 2b: Image Brightness Kernel
- **Image Size**: $512 \times 512 = 262,144$ pixels.
- **Kernel Mapping**: Each thread processes exactly 1 pixel:
  ```cuda
  __global__ void adjustBrightness(unsigned char *image, int width, int height, int offset) {
      int col = blockIdx.x * blockDim.x + threadIdx.x;
      int row = blockIdx.y * blockDim.y + threadIdx.y;
      if (col < width && row < height) {
          int idx = row * width + col;
          int val = image[idx] + offset;
          image[idx] = (unsigned char)(val > 255 ? 255 : (val < 0 ? 0 : val));
      }
  }
  ```
- **Verification**: Pixel intensity distribution shifted by $+50$, clamped at $255$. Mean intensity increased from $109.8$ to $159.8$.

### 5.3 Problem 2c: 2D Matrix Addition ($C = A + B$)
- **Matrix Dimension**: $1024 \times 1024$ ($1,048,576$ floating-point elements).
- **Execution Config**: `block(16, 16)`, `grid(64, 64)`.
- **Memory Transfer**:
  1. `cudaMalloc` allocations for device buffers $d_A, d_B, d_C$.
  2. `cudaMemcpy(..., cudaMemcpyHostToDevice)` inputs.
  3. `matrixAdd<<<grid, block>>>(d_A, d_B, d_C, 1024)`.
  4. `cudaMemcpy(..., cudaMemcpyDeviceToHost)` results.
- **Verification**: $100\%$ of $1,048,576$ elements verified against host CPU calculation ($|C_{i} - (A_{i} + B_{i})| < 10^{-6}$).

---

## 6. Compilation & Execution Guide

```bash
# On systems equipped with NVIDIA GPU & CUDA Toolkit (nvcc):
nvcc -O3 -arch=sm_75 device_query.cu -o device_query
./device_query

nvcc -O3 -arch=sm_75 threads_blocks.cu -o threads_blocks
./threads_blocks

nvcc -O3 -arch=sm_75 global_index_2d.cu -o global_index_2d
./global_index_2d

nvcc -O3 -arch=sm_75 brightness.cu -o brightness
./brightness

nvcc -O3 -arch=sm_75 matrix_add.cu -o matrix_add
./matrix_add
```
