# HPC Tutorial 2: Shared-Memory Parallel Programming with POSIX Threads (Pthreads)

## High Performance Computing Laboratory Report & Experimental Study

---

## 1. Executive Summary & Overview

POSIX Threads (Pthreads) is the standard IEEE shared-memory threading model for C/C++ in Unix/Linux and POSIX environments. This laboratory explores three foundational aspects of concurrent programming:
1. **Embarrassingly Parallel Numerical Simulation**: Monte Carlo estimation of $\pi$ with independent pseudorandom streams.
2. **Domain Decomposition Image Processing**: $3 \times 3$ Sobel convolution edge detection across large ($4096 \times 4096$) 2D image matrices.
3. **Deadlock Analysis and Prevention**: The classic Dining Philosophers synchronization problem, contrasting a naive circular-wait implementation against a strictly ordered, deadlock-free resource allocation hierarchy.

---

## 2. Directory Structure

```text
Tutorial-2/
├── plots/
│   ├── plot1_pi_vs_points.png        # Sample size N vs Estimated Pi
│   ├── plot2_error_vs_points.png      # Sample size N vs Absolute Error (Log-Log)
│   ├── plot3_time_vs_threads.png      # Thread count vs Monte Carlo execution time
│   ├── plot4_speedup_vs_threads.png   # Thread count vs Speedup (Monte Carlo)
│   ├── plot5_efficiency_vs_threads.png# Thread count vs Efficiency (Monte Carlo)
│   ├── plot6_edge_time_vs_threads.png # Thread count vs Edge Detection execution time
│   ├── plot7_edge_speedup_vs_threads.png # Thread count vs Speedup (Edge Detection)
│   └── plot8_edge_efficiency_vs_threads.png # Thread count vs Efficiency (Edge Detection)
├── q1_serial.c                       # Serial Monte Carlo Pi implementation
├── q1_parallel.c                     # Parallel Pthreads Monte Carlo Pi implementation
├── q2_serial.c                       # Serial Sobel 3x3 edge detection on 4096x4096 image
├── q2_parallel.c                     # Parallel Pthreads Sobel edge detection (1D row decomposition)
├── q3_dining.c                       # Dining Philosophers: Deadlock demonstration & Odd/Even resolution
├── 1Tutorial Questions.docx          # Original laboratory problem specification
├── Results.pdf                       # Comprehensive publication-quality PDF report
└── README.md                         # Full documentation & experimental evaluation
```

---

## 3. Problem 1: Monte Carlo Estimation of $\pi$

### 3.1 Mathematical Formulation
Consider a unit circle of radius $R$ inscribed within a square of side length $2R$:
- $\text{Area}_{\text{square}} = (2R)^2 = 4R^2$
- $\text{Area}_{\text{circle}} = \pi R^2$

The geometric probability $P$ that a uniformly distributed random point $(x, y) \in [-R, R] \times [-R, R]$ lies within the circle is:
$$P = \frac{\text{Area}_{\text{circle}}}{\text{Area}_{\text{square}}} = \frac{\pi R^2}{4R^2} = \frac{\pi}{4}$$

Rearranging gives the Monte Carlo estimator for sample size $N$:
$$\pi \approx 4 \times \frac{N_{\text{inside}}}{N}$$
where $N_{\text{inside}}$ satisfies $x^2 + y^2 \le R^2$.

### 3.2 Parallel Implementation Strategy
- **Thread Safety**: The standard C library `rand()` shares internal state across threads, introducing catastrophic serialization and contention. We employ a thread-safe pseudorandom generator (`rand_r` with a unique per-thread seed initialized to `seed = 42 + thread_id * 1000`).
- **Load Balancing**: Each of $T$ threads is assigned exactly $N / T$ iterations.
- **Reduction**: Each thread computes a private `local_count` without locks or shared variable updates during the compute loop. Partial counts are accumulated once upon `pthread_join()`, achieving zero synchronization overhead during execution.

### 3.3 Experimental Results & Scaling
Tested on Windows x64 across sample sizes $N \in [10, 10^9]$ and thread counts $T \in [1, 2, 4, 8, 16]$:

| Sample Size ($N$) | Threads ($T$) | Estimated $\pi$ | Absolute Error | Execution Time (s) | Speedup ($S$) | Efficiency ($E$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $10^1$ | 1 | 4.000000 | 0.858407 | 0.001506 | 1.00x | 100.0% |
| $10^2$ | 1 | 3.040000 | 0.101593 | 0.000200 | 1.00x | 100.0% |
| $10^4$ | 1 | 3.152800 | 0.011207 | 0.000278 | 1.00x | 100.0% |
| $10^6$ | 1 | 3.140852 | 0.000741 | 0.004495 | 1.00x | 100.0% |
| $10^7$ | 1 | 3.142466 | 0.000873 | 0.042829 | 1.00x | 100.0% |
| $10^8$ | 1 | 3.141367 | 0.000226 | 0.515155 | 1.00x | 100.0% |
| **$10^9$** | **1** | **3.141533** | **0.000059** | **5.300506** | **1.00x** | **100.0%** |
| $10^9$ | 2 | 3.141551 | 0.000041 | 2.308114 | 2.30x | 114.8% |
| $10^9$ | 4 | 3.141494 | 0.000098 | 1.434856 | 3.69x | 92.4% |
| $10^9$ | 8 | 3.141517 | 0.000076 | 1.127754 | 4.70x | 58.8% |
| $10^9$ | 16 | 3.141612 | 0.000020 | 0.983907 | 5.39x | 33.7% |

*Key Takeaway*: The statistical error follows the theoretical $O(1/\sqrt{N})$ curve, decreasing from $\approx 0.858$ at $N=10$ to $0.000019$ at $N=10^9$. Parallel speedup reaches $5.39\times$ on $N=10^9$.

---

## 4. Problem 2: Parallel Edge Detection Using Pthreads

### 4.1 Sobel Operator Convolution
For an $n \times n$ grayscale image, the $3 \times 3$ Sobel gradient masks are:
$$G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}$$

For each interior pixel $(r, c)$:
$$G = \min\left(255, \sqrt{G_x^2 + G_y^2}\right)$$
Computing each output pixel requires $9$ multiply-accumulate operations per channel.

### 4.2 1D Row Domain Decomposition
Rows $1$ through $n-2$ are partitioned across $T$ threads:
- `start_row = 1 + thread_id * (n - 2) / T`
- `end_row   = 1 + (thread_id + 1) * (n - 2) / T`
Because each thread reads from input rows `[start_row-1, end_row]` and writes only to output rows `[start_row, end_row-1]`, no write-write conflicts or thread locks are required.

### 4.3 Benchmarking on $4096 \times 4096$ Image (16.7 Megapixels)

| Configuration | Execution Time (s) | Speedup Factor | Parallel Efficiency |
|:---|:---:|:---:|:---:|
| **Serial Baseline** | 0.073147 s | 1.00x | 100.0% |
| **1 Pthread** | 0.052754 s | 1.00x | 100.0% |
| **2 Pthreads** | 0.022840 s | 2.31x | 115.5% |
| **4 Pthreads** | 0.017064 s | 3.09x | 77.3% |
| **8 Pthreads** | 0.015113 s | 3.49x | 43.6% |
| **16 Pthreads** | 0.013460 s | 3.92x | 24.5% |

---

## 5. Answers to Discussion Questions

### Q1: How does increasing the number of Pthreads affect execution time?
Execution time drops steeply when moving from 1 to 2 threads (0.0528s $\rightarrow$ 0.0228s, a $2.31\times$ reduction) and continues to decrease up to 16 threads (0.0135s). However, the marginal benefit flattens beyond the number of physical cores as memory bandwidth saturation and thread creation/joining overheads dominate.

### Q2: Does the speedup increase linearly with the number of threads? Explain.
No, speedup does not scale linearly indefinitely. While near-linear scaling is observed for $T=2$ ($S=2.31\times$) and $T=4$ ($S=3.09\times$), speedup sub-linearly plateaus at $3.92\times$ for $T=16$. In accordance with **Amdahl's Law**, memory bus contention (Sobel convolution is memory-bandwidth bound) and serial startup/shutdown components limit speedup.

### Q3: What factors limit the achievable speedup?
1. **Memory Bus Bandwidth**: In image convolution, reading $4096 \times 4096 \times 1$ byte and writing the output stresses the CPU cache and DRAM channels. Multiple cores compete for cache line fills.
2. **Thread Management Overhead**: `pthread_create()` and `pthread_join()` require kernel context switches and OS scheduling.
3. **Physical vs. Logical Cores**: Hyper-Threading shares execution pipelines and L1/L2 caches between sibling hardware threads, providing less than $2\times$ physical throughput.

### Q4: Why can efficiency decrease as the number of threads increases?
Parallel efficiency is defined as $E = \frac{S}{T} = \frac{T_1}{T \cdot T_T}$. As $T$ increases, the fixed overhead of thread management and memory bus contention is amortized over smaller row blocks per thread, causing idle CPU stall cycles. Consequently, efficiency falls from $115.5\%$ at $T=2$ to $24.5\%$ at $T=16$.

### Q5: Compare the performance of the serial and Pthreads implementations.
The serial implementation executes in $0.0731$s. The parallel Pthreads implementation with 4 threads executes in $0.0171$s ($4.28\times$ faster than serial) and with 16 threads in $0.0135$s ($5.43\times$ faster than serial). The parallel implementation achieves massive throughput improvements without sacrificing numerical precision.

---

## 6. Problem 3: Dining Philosophers Synchronization

### 6.1 Problem Formulation & Deadlock Condition
Five philosophers sit around a circular table. Five chopsticks (mutexes) are placed between them. Each philosopher alternates between Thinking and Eating. Eating requires acquiring both left (`id`) and right (`(id+1)%5`) chopsticks.

**Coffman Deadlock Conditions**:
1. *Mutual Exclusion*: Each chopstick can be held by only one philosopher.
2. *Hold and Wait*: A philosopher holds one chopstick while blocking to acquire the second.
3. *No Preemption*: A chopstick cannot be forcibly taken from a philosopher.
4. *Circular Wait*: Philosopher $0$ waits for $1$, $1$ waits for $2$, $2$ waits for $3$, $3$ waits for $4$, and $4$ waits for $0$.

### 6.2 Solution 1: Deadlock Demonstration
In `q3_dining.c`, each philosopher attempts to lock their left chopstick first. When all 5 philosophers execute simultaneously, each grabs their left chopstick and simultaneously attempts to acquire their right chopstick. A deadlock is immediately detected via non-blocking `pthread_mutex_trylock` timeout diagnostics.

### 6.3 Solution 2: Deadlock-Free Resource Ordering (Odd/Even Asymmetric Policy)
We break the **Circular Wait** condition by enforcing an asymmetric resource acquisition policy:
- **Odd Philosophers** ($1, 3$): Pick up **left** chopstick first, then **right**.
- **Even Philosophers** ($0, 2, 4$): Pick up **right** chopstick first, then **left**.

Because at least two adjacent philosophers compete for the same initial chopstick, circular waiting is mathematically impossible. All 5 philosophers eat and finish all scheduled meals safely without deadlock or starvation.

---

## 7. Compilation and Execution Guide

```bash
# Question 1: Monte Carlo Estimation of Pi
gcc -O3 -pthread q1_serial.c -o q1_serial -lm
gcc -O3 -pthread q1_parallel.c -o q1_parallel -lm
./q1_serial
./q1_parallel

# Question 2: Parallel Edge Detection
gcc -O3 -pthread q2_serial.c -o q2_serial -lm
gcc -O3 -pthread q2_parallel.c -o q2_parallel -lm
./q2_serial
./q2_parallel

# Question 3: Dining Philosophers
gcc -O3 -pthread q3_dining.c -o q3_dining
./q3_dining
```
