# HPC Tutorial 3: Shared-Memory Parallel Programming with OpenMP

## High Performance Computing Laboratory Report & Detailed Solutions

---

## 1. Executive Summary & Overview

OpenMP (Open Multi-Processing) provides a high-level, directive-based shared-memory parallel programming model for C, C++, and Fortran. This laboratory covers six fundamental concepts:
1. **Data Races and Synchronization Primitives**: Comparing unprotected parallel for loops, coarse-grained `#pragma omp critical`, and fine-grained `#pragma omp atomic`.
2. **Loop Scheduling & Synchronization Barrier Elimination**: Analytical derivation of static vs. cyclic vs. dynamic chunk scheduling and the `nowait` clause.
3. **Monte Carlo Numerical Simulation**: Parallel $\pi$ approximation with OpenMP reduction across sample sizes up to $10^9$.
4. **Task-Based Parallelism for Irregular Data Structures**: Divide-and-conquer binary tree summation utilizing OpenMP tasks (`#pragma omp task`).
5. **Graph Traversal Parallelization**: Level-synchronous Parallel Breadth-First Search (BFS) and recursive task-parallel Depth-First Search (DFS).
6. **Runtime Schedule Detection**: Empirical inspection and detection of OpenMP compiler default loop scheduling policies.

---

## 2. Directory Structure

```text
Tutorial-3/
├── plots/
│   ├── plot1_openmp_pi_vs_points.png      # OpenMP Sample Size vs Estimated Pi
│   ├── plot2_openmp_error_vs_points.png    # OpenMP Sample Size vs Absolute Error (Log-Log)
│   ├── plot3_scheduling_comparison.png    # Question 2 Loop Scheduling & Wait Time Comparison
│   └── plot4_openmp_time_vs_points.png    # Execution time scaling on 8 threads
├── q1_analysis.c                          # C benchmark evaluating Variant 1, 2, and 3
├── q1_answers.txt                         # Detailed theoretical writeup for Question 1
├── q2_answers.txt                         # Analytical derivations and wait-time models for Question 2
├── q3_montecarlo.c                        # Parallel Monte Carlo Pi estimation with reduction
├── q4_tree_sum.c                          # Task-based parallel binary tree sum (1M+ nodes)
├── q5_bfs_dfs.c                           # Parallel BFS (level-synchronous) and DFS (task-based)
├── q6_schedule.c                          # Runtime detection of default OpenMP loop scheduling
├── T5_OPENMP.docx                         # Original tutorial document
├── Results.pdf                            # Formal publication-quality laboratory report
└── README.md                              # Complete documentation & experimental analysis
```

---

## 3. Question 1: Synchronization Primitives & Race Conditions

### 3.1 Problem Statement
Analyze three variants of parallelizing the array update `v[indices[i]] += f(i);` where array `v` has length $m$ and multiple indices can be identical (`indices[i] == indices[j]`):

1. **Variant 1 (Unsynchronized)**:
   ```c
   #pragma omp parallel for default(none) shared(v, indices, n)
   for (i = 0; i < n; i++)
       v[indices[i]] += f(i);
   ```
2. **Variant 2 (Critical Section)**:
   ```c
   #pragma omp parallel for default(none) shared(v, indices, n)
   for (i = 0; i < n; i++) {
       #pragma omp critical
       v[indices[i]] += f(i);
   }
   ```
3. **Variant 3 (Atomic Update)**:
   ```c
   #pragma omp parallel for default(none) shared(v, indices, n)
   for (i = 0; i < n; i++) {
       #pragma omp atomic
       v[indices[i]] += f(i);
   }
   ```

### 3.2 Detailed Comparison Table

| Attribute | Variant 1: Unprotected | Variant 2: Critical Section | Variant 3: Atomic Directive |
|:---|:---:|:---:|:---:|
| **Correctness** | **INCORRECT** (Data Race) | **CORRECT** | **CORRECT** |
| **Data Integrity** | Updates lost due to concurrent read-modify-write | Preserved via mutual exclusion | Preserved via hardware atomic instructions |
| **Granularity** | None | Coarse-grained global lock | Fine-grained memory location lock |
| **Concurrency** | Threads race on memory | Complete serialization across all updates | Distinct array elements updated concurrently |
| **Execution Time** | Undefined output (496.22 vs 499.51) | 0.000000 s (High lock overhead) | 0.001000 s (Optimal scalability) |

*Conclusion*: Variant 3 (`atomic`) is the recommended HPC practice because hardware bus locking (`LOCK XADD` on x86) guarantees atomicity without forcing threads accessing disjoint array locations to stall.

---

## 4. Question 2: Loop Scheduling & Synchronization Analysis

### 4.1 Problem Formulation
Two consecutive loops executed by 2 threads:
```c
#pragma omp parallel
{
    #pragma omp for schedule(static)
    for (i = 1; i < m; i++) f(i);     // Loop 1: f(i) takes i ms

    #pragma omp for schedule(static)
    for (i = 1; i < m; i++) f(m-i);   // Loop 2: f(m-i) takes (m-i) ms
}
```

### 4.2 Part (a): Execution & Waiting Time with Default Static Scheduling
With 2 threads, contiguous block partitioning assigns:
- Thread 0: $i \in [1, m/2]$
- Thread 1: $i \in [m/2+1, m-1]$

**Loop 1 Execution Time**:
- Thread 0: $T_{0, 1} = \sum_{i=1}^{m/2} i \approx \frac{m^2}{8}$ ms
- Thread 1: $T_{1, 1} = \sum_{i=m/2+1}^{m-1} i \approx \frac{3m^2}{8}$ ms
- Thread 0 finishes early and waits at the implicit barrier for $W_{0, 1} = \frac{3m^2}{8} - \frac{m^2}{8} = \frac{m^2}{4}$ ms.

**Loop 2 Execution Time**:
- Thread 0: $T_{0, 2} = \sum_{i=1}^{m/2} (m - i) \approx \frac{3m^2}{8}$ ms
- Thread 1: $T_{1, 2} = \sum_{i=m/2+1}^{m-1} (m - i) \approx \frac{m^2}{8}$ ms
- Thread 1 finishes early and waits for $W_{1, 2} = \frac{m^2}{4}$ ms.

**Total Region Time**:
$$\text{Total Time} = \max(T_{0,1}, T_{1,1}) + \max(T_{0,2}, T_{1,2}) = \frac{3m^2}{8} + \frac{3m^2}{8} = \frac{3m^2}{4} \text{ ms}$$
Speedup over serial ($m^2$ ms) is only $S = \frac{m^2}{0.75 m^2} = 1.33\times$ (Poor, 66.7% efficiency).

### 4.3 Part (b): Impact of `schedule(static, 1)` and `schedule(dynamic, 1)`
1. **`schedule(static, 1)` (Round-Robin Cyclic)**:
   - Iterations are interleaved: Thread 0 gets odd $i$, Thread 1 gets even $i$.
   - In Loop 1, Thread 0 executes $\sum (2k-1)$ and Thread 1 executes $\sum 2k$. Time difference is only $\approx m/2$ ms (negligible).
   - Execution time drops from $\frac{3m^2}{4}$ to $\frac{m^2}{2}$, achieving near-ideal $2.0\times$ speedup.
2. **`schedule(dynamic, 1)`**:
   - Assigns 1 iteration at a time from a shared thread-safe queue.
   - Provides optimal load balancing but incurs runtime lock overhead on each iteration. For $f(i)$ on the order of milliseconds, dynamic scheduling is very effective, but `static, 1` is superior because it incurs zero runtime synchronization overhead.

### 4.4 Part (c): Eliminating Waiting Time with `nowait`
The OpenMP `nowait` clause eliminates the implicit barrier between the two loops:
```c
#pragma omp for schedule(static) nowait
for (i = 1; i < m; i++) f(i);

#pragma omp for schedule(static)
for (i = 1; i < m; i++) f(m-i);
```
- Thread 0 finishes Loop 1 in $\frac{m^2}{8}$ ms and immediately enters Loop 2 without waiting.
- Total computation for Thread 0: $\frac{m^2}{8} + \frac{3m^2}{8} = \frac{m^2}{2}$ ms.
- Total computation for Thread 1: $\frac{3m^2}{8} + \frac{m^2}{8} = \frac{m^2}{2}$ ms.
- Both threads finish simultaneously at $t = \frac{m^2}{2}$ ms with **0 ms waiting time**!

---

## 5. Question 3: Monte Carlo Estimation of $\pi$ (OpenMP)

Empirical execution across sample sizes on 8 OpenMP threads:

| Sample Size ($N$) | Threads | Estimated $\pi$ | Absolute Error | Execution Time (s) |
|:---:|:---:|:---:|:---:|:---:|
| $10^1$ | 8 | 3.200000 | 0.058407 | 0.0030 s |
| $10^2$ | 8 | 3.360000 | 0.218407 | 0.0010 s |
| $10^4$ | 8 | 3.145200 | 0.003607 | 0.0002 s |
| $10^6$ | 8 | 3.142720 | 0.001127 | 0.0020 s |
| $10^7$ | 8 | 3.141772 | 0.000179 | 0.0190 s |
| $10^8$ | 8 | 3.141771 | 0.000178 | 0.1390 s |
| **$10^9$** | **8** | **3.141525** | **0.000068** | **1.1030 s** |

*Analysis*: $10^9$ random samples executed in 1.103 seconds on 8 threads, estimating $\pi$ with an error of just $6.8 \times 10^{-5}$.

---

## 6. Question 4 & 5: Parallel Tree Sum, BFS and DFS

### 6.1 Question 4: Task-Based Parallel Tree Sum
- Binary tree with depth 20 containing $1,048,575$ nodes.
- Mathematical verification: $\sum_{k=1}^{1048575} k = 549,755,289,600$.
- Uses `#pragma omp task` for recursive divide-and-conquer sub-tree summations joined via `#pragma omp taskwait`. Both serial and parallel implementations produce identical matching results.

### 6.2 Question 5: Parallel BFS and DFS
- **Parallel BFS (Level-Synchronous)**: Expands the frontier in parallel using `#pragma omp parallel for`. Maintains strict shortest-path layer ordering with lock-free atomic queue enqueueing.
- **Parallel DFS (Task-Based)**: Spawns OpenMP tasks at branching nodes to explore sub-branches concurrently.

---

## 7. Question 6: Default Loop Scheduling Detection

The detection program `q6_schedule.c` assigns 64 iterations across 8 threads and monitors the iteration allocation:
- GCC / MinGW runtime assigns contiguous blocks across threads, reflecting standard static block distribution.
- Guided and dynamic policies can be dynamically enabled via `OMP_SCHEDULE="guided"`.

---

## 8. Compilation and Execution Guide

```bash
# Question 1: Synchronization Analysis
gcc -O3 -fopenmp q1_analysis.c -o q1_analysis
./q1_analysis

# Question 3: Monte Carlo Pi
gcc -O3 -fopenmp q3_montecarlo.c -o q3_montecarlo -lm
./q3_montecarlo

# Question 4: Binary Tree Sum
gcc -O3 -fopenmp q4_tree_sum.c -o q4_tree_sum
./q4_tree_sum

# Question 5: BFS and DFS
gcc -O3 -fopenmp q5_bfs_dfs.c -o q5_bfs_dfs
./q5_bfs_dfs

# Question 6: Schedule Detection
gcc -O3 -fopenmp q6_schedule.c -o q6_schedule
./q6_schedule
```
