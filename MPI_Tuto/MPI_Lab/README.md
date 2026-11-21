# Distributed Matrix Multiplication using MPI (Block-Cyclic Row Distribution)

## 1. Overview & Problem Statement

This project implements **Distributed Matrix Multiplication ($C = A \times B$)** using the Message Passing Interface (MPI) under the distributed computing paradigm.

### Prompt Requirements:
- **Matrix Generation**: Process 0 generates integer arrays for matrices $A$ and $B$.
- **Data Distribution**: Process 0 distributes rows of Matrix $A$ to all processes in a **block-cyclic** fashion with a block size of $B = 500$ rows.
- **Independent Computation**: Each process computes its assigned independent rows of Matrix $C$.
- **Result Collection**: Process 0 collects the computed rows and displays the final matrix result.
- **Required MPI Functions**: `MPI_Scatter()`, `MPI_Gather()`, and `MPI_Bcast()`.
- **Design Philosophy**: Easy to explain and beginner-friendly, specifically addressing how to handle columns when only row distribution is specified.

---

## 2. Architectural Design & Pedagogical Explanation

### Why 1D Row Distribution? (Addressing the "Column Not Given" Note)
In general matrix multiplication:
$$C_{i, j} = \sum_{k=0}^{N-1} A_{i, k} \times B_{k, j}$$

Notice that computing any single element $C_{i, j}$ depends **only** on:
1. **Row $i$ of Matrix $A$**
2. **Column $j$ of Matrix $B$**

Therefore, to compute the entire **Row $i$ of Matrix $C$**, a process only requires **Row $i$ of Matrix $A$** and the **entire Matrix $B$**. It does **not** need any other rows of $A$ or other rows of $C$!

```
                  Matrix B (N x N)
                   [BroadCast]
                  +---+---+---+
                  |   |   |   |
                  |   |   |   |
                  +---+---+---+
                        |
                        v
Matrix A (N x N)              Matrix C (N x N)
+------------------+          +------------------+
| Block 0 (Rows)   | ------>  | Block 0 of C     |  --> Process 0
+------------------+          +------------------+
| Block 1 (Rows)   | ------>  | Block 1 of C     |  --> Process 1
+------------------+          +------------------+
| Block 2 (Rows)   | ------>  | Block 2 of C     |  --> Process 0 (Round 2)
+------------------+          +------------------+
| Block 3 (Rows)   | ------>  | Block 3 of C     |  --> Process 1 (Round 2)
+------------------+          +------------------+
 [MPI_Scatter]                 [MPI_Gather]
```

By broadcasting Matrix $B$ using `MPI_Bcast()`, every process has all columns of $B$. Consequently, we do **not** need to partition columns or perform complex inter-process communications during the computation! This produces a clean, distributed computing architecture that is simple for beginners to grasp and debug.

---

### What is Block-Cyclic Distribution?
Instead of giving each process one large contiguous block of rows (pure block distribution), **block-cyclic** distribution partitions rows into fixed chunks of size $B$ (e.g. 500 rows) and deals them out round-robin like playing cards:

Suppose $N = 2000$, Block Size $B = 500$, and Process Count $P = 2$:
- Total Blocks = $2000 / 500 = 4$ blocks:
  - **Round 0**:
    - Process 0 receives **Block 0** (Rows $0 \dots 499$)
    - Process 1 receives **Block 1** (Rows $500 \dots 999$)
  - **Round 1**:
    - Process 0 receives **Block 2** (Rows $1000 \dots 1499$)
    - Process 1 receives **Block 3** (Rows $1500 \dots 1999$)

### How the Required MPI Primitives are Applied:
1. **`MPI_Bcast()`**:
   ```c
   MPI_Bcast(full_B, N * N, MPI_INT, 0, MPI_COMM_WORLD);
   ```
   Process 0 transmits the complete Matrix $B$ to every process in a single collective call.
2. **`MPI_Scatter()`**:
   ```c
   for (int r = 0; r < num_rounds; r++) {
       MPI_Scatter(&A[r * P * B * N], B * N, MPI_INT,
                   &local_A[r * B * N], B * N, MPI_INT,
                   0, MPI_COMM_WORLD);
   }
   ```
   In each round $r$, `MPI_Scatter()` takes $P$ blocks from Matrix $A$ and sends exactly 1 block of $B$ rows ($B \times N$ integers) to each process.
3. **Independent Computation**:
   ```c
   for (int i = 0; i < local_rows; i++) {
       for (int j = 0; j < N; j++) {
           long long sum = 0;
           for (int k = 0; k < N; k++) {
               sum += local_A[i * N + k] * full_B[k * N + j];
           }
           local_C[i * N + j] = sum;
       }
   }
   ```
   Zero inter-process communication occurs here. Each process computes its local rows in parallel!
4. **`MPI_Gather()`**:
   ```c
   for (int r = 0; r < num_rounds; r++) {
       MPI_Gather(&local_C[r * B * N], B * N, MPI_INT,
                  &C[r * P * B * N], B * N, MPI_INT,
                  0, MPI_COMM_WORLD);
   }
   ```
   Process 0 collects each computed block back in cyclic order, assembling the final result matrix $C$.

---

## 3. Included Programs

| File | Description | Recommended Usage |
| :--- | :--- | :--- |
| `matrix_mult_block_cyclic.c` | Production program with block size $B = 500$, scalable to $N = 1000, 2000$, with timing and automated verification. | Benchmarking and large matrix experiments ($n \sim 10^6, 10^8$). |
| `matrix_mult_beginner_demo.c` | Self-contained, step-by-step educational demo ($N = 8$, block size = 2) that prints every row, received chunk, and intermediate result. | Teaching, class presentations, and beginner debugging. |

---

## 4. Compilation & Execution Commands

### Compilation:
Open your shell in the `MPI_Lab` directory:
```powershell
# Compile the main scalable program
mpicc -O3 matrix_mult_block_cyclic.c -o matrix_mult_block_cyclic.exe

# Compile the beginner demo program
mpicc matrix_mult_beginner_demo.c -o matrix_mult_beginner_demo.exe
```

### Execution:

#### Running the Beginner Demo (8x8 matrix, 2 processes):
```powershell
mpiexec -n 2 .\matrix_mult_beginner_demo.exe
```

#### Running the Main Program (Default: N = 1000, Block Size = 500, 2 processes):
```powershell
mpiexec -n 2 .\matrix_mult_block_cyclic.exe 1000 500
```

#### Running with 4 processes (N = 2000, Block Size = 500):
```powershell
mpiexec -n 4 .\matrix_mult_block_cyclic.exe 2000 500
```

#### Running with Multiple Rounds (N = 1000, Block Size = 250, 2 rounds):
```powershell
mpiexec -n 2 .\matrix_mult_block_cyclic.exe 1000 250
```

---

## 5. Sample Outputs

### 1. Beginner Demo Output (`mpiexec -n 2 .\matrix_mult_beginner_demo.exe`):
```text
======================================================================
 BEGINNER'S STEP-BY-STEP MPI MATRIX MULTIPLICATION DEMO
 Matrix Size: 8x8 | Processes: 2 | Block Size: 2 rows
======================================================================

--- Process 0 Generated Matrix A ---
Row 0: [    1    1    1    1    1    1    1    1 ]
Row 1: [    2    2    2    2    2    2    2    2 ]
Row 2: [    3    3    3    3    3    3    3    3 ]
Row 3: [    4    4    4    4    4    4    4    4 ]
Row 4: [    5    5    5    5    5    5    5    5 ]
Row 5: [    6    6    6    6    6    6    6    6 ]
Row 6: [    7    7    7    7    7    7    7    7 ]
Row 7: [    8    8    8    8    8    8    8    8 ]

--- Process 0 Generated Matrix B ---
Row 0: [    2    1    1    1    1    1    1    1 ]
Row 1: [    1    2    1    1    1    1    1    1 ]
... (Identity diagonal + 1s)

----------------------------------------------------------------------
 STEP 1: Broadcasting Matrix B to ALL processes via MPI_Bcast()...
----------------------------------------------------------------------
[Process 1] Successfully received full Matrix B via MPI_Bcast()!

----------------------------------------------------------------------
 STEP 2: Distributing Matrix A rows in BLOCK-CYCLIC fashion via MPI_Scatter()...
         Each process gets 2 rows per round across 2 round(s).
----------------------------------------------------------------------
[Process 0] Round 0: Received Block (Global Rows 0 and 1)
[Process 1] Round 0: Received Block (Global Rows 2 and 3)
[Process 0] Round 1: Received Block (Global Rows 4 and 5)
[Process 1] Round 1: Received Block (Global Rows 6 and 7)

----------------------------------------------------------------------
 STEP 3: Each process independently computes its assigned rows (C = A x B)...
----------------------------------------------------------------------
[Process 0] Finished computing its 4 assigned rows!
[Process 1] Finished computing its 4 assigned rows!

----------------------------------------------------------------------
 STEP 4: Process 0 gathers computed blocks from all processes via MPI_Gather()...
----------------------------------------------------------------------

======================================================================
 STEP 5: Process 0 Displays the Final Gathered Matrix C Result
======================================================================
--- Final Gathered Matrix C = A x B ---
Row 0: [    9    9    9    9    9    9    9    9 ]
Row 1: [   18   18   18   18   18   18   18   18 ]
Row 2: [   27   27   27   27   27   27   27   27 ]
Row 3: [   36   36   36   36   36   36   36   36 ]
Row 4: [   45   45   45   45   45   45   45   45 ]
Row 5: [   54   54   54   54   54   54   54   54 ]
Row 6: [   63   63   63   63   63   63   63   63 ]
Row 7: [   72   72   72   72   72   72   72   72 ]

[VERIFICATION SUCCESS] Row 0 values are all 9 as mathematically expected!
======================================================================
```

### 2. Scalable Benchmark Output (`mpiexec -n 2 .\matrix_mult_block_cyclic.exe 1000 500`):
```text
================================================================================
 MPI Distributed Matrix Multiplication (Block-Cyclic Row Distribution)
================================================================================
 Matrix Dimension (N x N)    : 1000 x 1000
 Total Matrix Elements (n)   : 1000000 (~10^6 elements)
 Number of MPI Processes     : 2
 Block Size (rows per block) : 500 rows
 Number of Cyclic Rounds     : 1 round(s)
 Rows Assigned per Process   : 500 rows (1 blocks)
 Communication Strategy      : 1D Row Block-Cyclic (MPI_Scatter, MPI_Bcast, MPI_Gather)
================================================================================
...
Computation & Communication Completed Successfully!
Total Elapsed Parallel Time: 0.7411 seconds
Estimated Performance      : 2.70 GFLOP/s

Verifying distributed computation against sequential reference...
[VERIFICATION PASSED] Result matrix C matches exact mathematical multiplication!
================================================================================
```

---

## 6. Verification and Correctness
Both programs include automated self-verification:
- The beginner demo mathematically proves that Row 0 matches $1 \times 2 + 7 \times 1 = 9$.
- The scalable program samples rows and computes a sequential CPU ground-truth check ($O(N^2)$ sample verification) ensuring that distributed calculations match exact mathematical results with zero numerical error.

