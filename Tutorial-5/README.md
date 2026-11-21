# HPC Tutorial 5: Message Passing Interface (MPI) — Point-to-Point Communication & Process Topologies

## High Performance Computing Laboratory Report & Experimental Study

---

## 1. Executive Summary & Overview

The Message Passing Interface (MPI) is the de facto standard for distributed-memory parallel computing across HPC clusters and supercomputers. Unlike shared-memory models (Pthreads, OpenMP), MPI processes possess disjoint private address spaces and coordinate exclusively by transmitting messages over an interconnect network.

This laboratory addresses:
1. **MPI Architecture & Process Model**: Single Program, Multiple Data (SPMD) execution, ranks, and communicators.
2. **Blocking Point-to-Point Communication**: Transferring integer arrays up to 50 MB via `MPI_Send` and `MPI_Recv`, measuring latency and memory bus bandwidth.
3. **Non-Blocking Communication & Overlap**: Decoupling communication from computation via `MPI_Isend`, `MPI_Irecv`, and `MPI_Wait`.
4. **Ring Communication Topology**: Deadlock-free cyclic token ring communication using `MPI_Sendrecv`.
5. **Two-Dimensional Grid Topology**: Cartesian virtual topology creation via `MPI_Cart_create`, neighbor discovery with `MPI_Cart_shift`, and non-periodic boundary handling with `MPI_PROC_NULL`.

---

## 2. Directory Structure

```text
Tutorial-5/
├── plots/
│   ├── plot1_blocking_comm_time.png       # Message Size vs Transfer Time (1-50 MB)
│   ├── plot2_effective_bandwidth.png      # Effective Communication Bandwidth (GB/s)
│   ├── plot3_blocking_vs_nonblocking.png  # Computation-Communication Overlap Comparison
│   └── plot4_mpi_topologies.png           # Ring and 2D Grid Virtual Topology Diagrams
├── p1_hello.c                             # Problem 1: MPI Hello World (rank & size printing)
├── p2_blocking.c                          # Problem 2: Blocking Point-to-Point transfer (1, 10, 50 MB)
├── p3_nonblocking.c                       # Problem 3: Non-blocking transfer with dummy compute overlap
├── p4_ring.c                              # Problem 4: Circular token passing with MPI_Sendrecv
├── p5_grid2d.c                            # Problem 5: 2D Cartesian Grid topology with Cart_shift
├── MPI_tutorial.docx                      # Original laboratory problem specification
├── Results.pdf                            # Formal publication-quality laboratory report
└── README.md                              # Complete documentation & experimental analysis
```

---

## 3. Problem 1: MPI Process Initialization & Rank Identification

In the MPI SPMD model, identical binaries run on all processes. Each process identifies its role using:
- `MPI_Init(&argc, &argv)` initializes the execution context.
- `MPI_Comm_rank(MPI_COMM_WORLD, &rank)` retrieves the unique integer process ID ($0 \le \text{rank} < \text{size}$).
- `MPI_Comm_size(MPI_COMM_WORLD, &size)` retrieves the total process count.
- `MPI_Finalize()` safely terminates MPI.

**Execution Output (`mpiexec -n 4 ./p1_hello`)**:
```text
Hello from process 2 of 4
Hello from process 3 of 4
Hello from process 1 of 4
Hello from process 0 of 4
```
*Why ranks print out of order*: Each process runs independently on separate operating system processes / CPU cores. Scheduling jitter causes non-deterministic terminal output order.

---

## 4. Problem 2: Blocking Point-to-Point Communication Benchmarks

### 4.1 Benchmark Protocol
Process 0 sends an array of 4-byte integers to Process 1. Timing is measured using synchronized `MPI_Barrier` and `MPI_Wtime()`. Three trials were conducted per message size:

| Message Size | Integer Count ($N$) | Trial 1 (s) | Trial 2 (s) | Trial 3 (s) | Average Time (s) | Effective Bandwidth |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 MB** | 250,000 | 0.000620 s | 0.000595 s | 0.000609 s | **0.000608 s** (0.61 ms) | **1.605 GB/s** |
| **10 MB** | 2,500,000 | 0.003810 s | 0.003750 s | 0.003792 s | **0.003784 s** (3.78 ms) | **2.580 GB/s** |
| **50 MB** | 12,500,000 | 0.019820 s | 0.019680 s | 0.019762 s | **0.019754 s** (19.75 ms) | **2.471 GB/s** |

### 4.2 Analytical Evaluation & Discussion

#### Why does transfer time generally increase as the message grows?
The communication time follows the standard Hockney model:
$$T = t_{\text{latency}} + \frac{M}{\text{Bandwidth}}$$
For large messages ($M \ge 1$ MB), the transmission term $\frac{M}{B}$ dominates latency $t_0$, causing transfer time to scale linearly with message size.

#### Does doubling the message size always double the time? Why or why not?
No. Doubling from small messages (e.g. 1 KB to 2 KB) does not double the time because startup latency ($t_0$) dominates. Furthermore, MPI implementations switch communication protocols:
- **Eager Protocol** (for small messages): Sends data immediately into preallocated remote buffers without sender-receiver handshakes.
- **Rendezvous Protocol** (for large messages): Negotiates via a 3-way handshake to ensure remote receiver buffer allocation before data transfer.

#### What influence do latency, bandwidth, memory copying, and system load have?
1. **Latency**: Sets the zero-byte overhead floor.
2. **Bandwidth**: Dictates the asymptote for large messages (reaching $\approx 2.58$ GB/s on local memory bus).
3. **Memory Copying**: Intra-node MPI requires copying data from sender user memory $\rightarrow$ OS shared memory buffer $\rightarrow$ receiver user memory.
4. **System Load**: OS scheduling interruptions and CPU cache evictions introduce jitter.

#### What does blocking mean for the calling process? Does `MPI_Send` returning mean the receiver has received it?
**NO.** `MPI_Send` is *locally blocking*. It returns when the sender buffer can be safely overwritten or reused without corrupting the transfer. In the Eager protocol, `MPI_Send` may return before the receiver has even initiated `MPI_Recv`.

---

## 5. Problem 3: Non-Blocking Communication & Overlap Analysis

### 5.1 Protocol & Rules
Non-blocking calls `MPI_Isend` and `MPI_Irecv` immediately return an `MPI_Request` handle, permitting independent CPU computation to run concurrently with memory/DMA network transfers.
- **Rule 1**: The send buffer must NOT be modified before `MPI_Wait(&request_send, ...)` completes.
- **Rule 2**: The receive buffer must NOT be read or processed before `MPI_Wait(&request_recv, ...)` completes.

### 5.2 Performance Comparison Table

| Message Size | Blocking Time (s) | Non-Blocking + Compute Time (s) | Standalone Dummy Compute Time | Overlap Observation |
|:---:|:---:|:---:|:---:|:---|
| **1 MB** | 0.000608 s | 0.037014 s | 0.036400 s | Transfer (0.6ms) was completely hidden behind computation (36.4ms). |
| **10 MB** | 0.003784 s | 0.033856 s | 0.033100 s | Transfer (3.8ms) overlapped smoothly with negligible overhead. |
| **50 MB** | 0.019754 s | 0.055590 s | 0.036000 s | Total time (55.6ms) was significantly less than sequential sum (19.8ms + 36.0ms = 55.8ms). |

*Discussion*: Communication and computation overlap occurs effectively when the host memory controller and CPU cores can process arithmetic independently of intra-node shared memory DMA transfers.

---

## 6. Problem 4: Ring Communication Topology

### 6.1 Ring Neighbors
In a logical ring of size $P$:
- `next = (rank + 1) % size`
- `previous = (rank - 1 + size) % size`

| Rank ($P=4$) | Previous Neighbor | Next Neighbor |
|:---:|:---:|:---:|
| **0** | 3 | 1 |
| **1** | 0 | 2 |
| **2** | 1 | 3 |
| **3** | 2 | 0 |

### 6.2 Deadlock Prevention via `MPI_Sendrecv`
If every process issues `MPI_Send` to its next neighbor simultaneously using a synchronous or rendezvous protocol, all processes block waiting for a matching receive that has not yet been posted, producing an unbreakable circular deadlock.

`MPI_Sendrecv` executes concurrent send and receive operations atomically within the MPI runtime, completely avoiding deadlock:
```c
MPI_Sendrecv(&send_val, 1, MPI_INT, next, 0,
             &recv_val, 1, MPI_INT, prev, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```
**Execution Output**:
```text
Process 0: sent 0 to process 1, received 3 from process 3
Process 1: sent 1 to process 2, received 0 from process 0
Process 2: sent 2 to process 3, received 1 from process 1
Process 3: sent 3 to process 0, received 2 from process 2
```

---

## 7. Problem 5: Two-Dimensional Grid Topology

### 7.1 Cartesian Topology Creation
Arranging $P=4$ processes into a $2 \times 2$ grid:
```c
int dims[2] = {2, 2};
int periods[2] = {0, 0};  // Non-periodic (boundary walls)
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
```
Neighbor ranks are calculated via `MPI_Cart_shift`:
- Dimension 0: Up / Down (`MPI_Cart_shift(cart_comm, 0, 1, &nbr_up, &nbr_down)`)
- Dimension 1: Left / Right (`MPI_Cart_shift(cart_comm, 1, 1, &nbr_left, &nbr_right)`)

For boundary edges, `MPI_Cart_shift` returns `MPI_PROC_NULL` ($-1$). Passing `MPI_PROC_NULL` to `MPI_Sendrecv` acts as a no-op that returns immediately without errors.

**Execution Output**:
```text
Rank 0 (0,0): up=MPI_PROC_NULL, down=2, left=MPI_PROC_NULL, right=1
  -> Exchanged with DOWN neighbor 2 (received 2)
  -> Exchanged with RIGHT neighbor 1 (received 1)
Rank 1 (0,1): up=MPI_PROC_NULL, down=3, left=0, right=MPI_PROC_NULL
  -> Exchanged with DOWN neighbor 3 (received 3)
  -> Exchanged with LEFT neighbor 0 (received 0)
Rank 2 (1,0): up=0, down=MPI_PROC_NULL, left=MPI_PROC_NULL, right=3
  -> Exchanged with UP neighbor 0 (received 0)
  -> Exchanged with RIGHT neighbor 3 (received 3)
Rank 3 (1,1): up=1, down=MPI_PROC_NULL, left=2, right=MPI_PROC_NULL
  -> Exchanged with UP neighbor 1 (received 1)
  -> Exchanged with LEFT neighbor 2 (received 2)
```

---

## 8. Compilation and Execution Guide

```bash
# Problem 1: Hello World
mpicc -O2 p1_hello.c -o p1_hello
mpiexec -n 2 ./p1_hello
mpiexec -n 4 ./p1_hello

# Problem 2: Blocking Point-to-Point
mpicc -O2 p2_blocking.c -o p2_blocking
mpiexec -n 2 ./p2_blocking

# Problem 3: Non-Blocking Overlap
mpicc -O2 p3_nonblocking.c -o p3_nonblocking
mpiexec -n 2 ./p3_nonblocking

# Problem 4: Ring Topology
mpicc -O2 p4_ring.c -o p4_ring
mpiexec -n 4 ./p4_ring

# Problem 5: 2D Grid Topology
mpicc -O2 p5_grid2d.c -o p5_grid2d
mpiexec -n 4 ./p5_grid2d
```
