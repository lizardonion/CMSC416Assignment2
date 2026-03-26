# A2 ↔ Midterm Topic Connections

This document connects the CMSC416 Assignment 2 code directly to the topics you listed for the midterm.

---

## 1. MPI Basics and Conventions

## Where this appears in the code

### `mpi_hello.c`
Best minimal example of:
- `MPI_Init`
- `MPI_Comm_rank`
- `MPI_Comm_size`
- `MPI_Finalize`
- root-only behavior with `if (proc_id == 0)`

### `heat_mpi.c` and `kmeans_mpi.c`
Show realistic use of the same basics inside actual algorithms.

## What to know

### Rank
A rank is a process ID inside a communicator.

### Size
The size is the total number of ranks in that communicator.

### Communicator
A communicator defines the group of processes participating in communication.
This assignment uses `MPI_COMM_WORLD` only.

### Root
A designated process, usually rank 0, often handles:
- input
- output
- orchestration
- centralized decisions

### SPMD
Every rank runs the same executable but behaves differently depending on its rank.

## Why it matters in A2

The assignment is fundamentally about moving from:
- one process that owns all data

to:
- many processes that each own only part of the data and must coordinate explicitly

---

## 2. Collective Communication Operations and how they are facilitated in MPI

Collective communication means **all processes in a communicator participate in the same communication call**.
MPI provides built-in, highly standardized operations for common communication patterns.

## A2 examples

### `MPI_Bcast`
Used in `kmeans_mpi.c`.

Broadcasts:
- problem metadata (`nclust`, `MAXITER`, `ndata`, `dim`)
- updated centroid coordinates

Why it fits:
- one process has information
- every process needs the same information

### `MPI_Scatterv`
Used in `kmeans_mpi.c`.

Distributes the input feature matrix from root to all ranks.

Why it fits:
- root initially owns the dataset
- work is partitioned by points
- partitions may differ in size, so the variable-size version is needed

### `MPI_Reduce`
Used in `kmeans_mpi.c`.

Combines:
- local partial centroid sums
- local cluster counts

Why it fits:
- every rank contributes a partial answer
- only root needs the final combined result for centroid recomputation/reporting

### `MPI_Allreduce`
Used in `kmeans_mpi.c`.

Combines local assignment-change counts into one global `nchanges` value and sends that result back to all processes.

Why it fits:
- every rank must know whether the algorithm has converged

### `MPI_Gather` / `MPI_Gatherv`
Used in both MPI problems.

- `heat_mpi.c` uses `MPI_Gather` to collect evenly sized local rod histories
- `kmeans_mpi.c` uses `MPI_Gatherv` to collect potentially uneven assignment arrays

Why it fits:
- local pieces must be reassembled at root

## Conceptual takeaway

Collectives are “facilitated” in MPI because the library already implements these standard patterns efficiently and safely.
You do not have to manually code every pairwise send/receive.
Instead, you choose the collective that matches the data flow.

---

## 3. Uses of MPI in Assignment 2 problems

## Problem 1: MPI Heat

Uses MPI for:
- process startup and rank discovery
- neighbor communication with `MPI_Sendrecv`
- final result collection with `MPI_Gather`

The communication is **local and structured**.
Only adjacent ranks exchange data.

## Problem 3: MPI K-means

Uses MPI for:
- metadata broadcast
- dataset distribution
- reduction of partial sums/counts
- global convergence detection
- final result collection

The communication is more **global and collective** than in heat.

## Comparison

### Heat
- mostly point-to-point communication
- nearest-neighbor pattern
- communication every timestep

### K-means
- mostly collective communication
- global synchronization between iterations
- more computation per communication phase

That contrast is very likely exam material.

---

## 4. Effective parallelization of the problems in A2

## Heat parallelization

### What was parallelized?
The rod positions.

### Why that works
Each update depends only on neighboring positions from the previous timestep.
So the rod can be split into contiguous chunks with halo exchange.

### Strengths
- intuitive ownership
- communication volume is small per exchange
- only adjacent ranks communicate

### Weaknesses
- communication happens every timestep
- if each process owns too little work, communication dominates
- strong scaling may be poor

## K-means parallelization

### What was parallelized?
The input points.

### Why that works
For a fixed set of centroids, each point can be handled independently.
That creates a lot of local work per process.

### Strengths
- natural load distribution
- large amount of local computation
- collectives occur only at phase boundaries

### Weaknesses
- synchronization each iteration
- root becomes a central point for I/O and centroid formation
- poor centroid balance can still affect efficiency somewhat

## Big exam idea

A parallelization is effective when the decomposition matches the algorithm’s dependency structure.

In A2:
- Heat → local-neighbor decomposition
- K-means → data-parallel point decomposition

---

## 5. Dense matrix multiplication parallel algorithms as discussed in lecture

Even though A2 does not implement matrix multiplication, its ideas transfer directly.

## Conceptual bridge from A2

### From heat
You learn:
- partitioning structured numerical data
- managing boundary dependencies
- exchanging only the needed edge information

That mindset is useful for block-based matrix algorithms.

### From K-means
You learn:
- distributing large inputs
- combining partial results with reductions
- broadcasting shared state

That mindset is useful for row-wise or block-wise matrix multiplication.

## Typical dense MM strategies

### Row-wise decomposition
Each process owns some rows of `A`.
All processes need `B`.
Then:
- scatter rows of `A`
- broadcast `B`
- compute local rows of `C`
- gather rows of `C`

This is structurally similar to K-means:
- distribute local work
- broadcast shared data
- gather outputs

### 2D block decomposition
Each process owns a block of `A` and `B`.
Communication is more advanced but improves scalability.

This is conceptually closer to more structured distributed numerical algorithms.

## What to say on an exam

You can say A2 prepares you for matrix multiplication by teaching:
- decomposition choices
- communication/computation tradeoffs
- when to broadcast shared data
- when to gather final outputs
- how algorithm structure determines the best MPI pattern

---

## 6. LU factorization and parallelizing it as discussed in lecture

LU factorization has a different dependency structure from A2, but the parallel reasoning is similar.

## Core LU idea

At each pivot step:
- determine pivot information
- update rows/blocks beneath or beyond the pivot
- continue to the next stage

## Parallel LU patterns that rhyme with A2

### Broadcast of shared stage data
Like broadcasting centroids in K-means, LU often requires broadcasting a pivot row, pivot block, or panel.

### Local updates after synchronization
Like heat updates or local point reassignment, each process updates its owned data using the newest shared information.

### Stage-by-stage dependency
Unlike K-means, LU has stronger sequential structure.
You cannot fully parallelize across pivot stages because later stages depend on earlier ones.

## Heat vs K-means vs LU

### Heat
- local dependency
- nearest-neighbor exchange
- many small communication steps

### K-means
- mostly independent local work per iteration
- global collectives at phase boundaries

### LU
- staged dependency
- repeated broadcasts and synchronized block updates
- less embarrassingly parallel than K-means

## What A2 gives you for LU

A2 trains the habit of asking:
- what data is local?
- what data is shared?
- when must processes synchronize?
- which communication primitive matches the need?

That is exactly the right mental framework for LU questions.

---

## 7. One-line mappings from files to topics

- `mpi_hello.c` → MPI startup, rank/size, root convention
- `heat_serial.c` → serial dependency analysis before parallelization
- `heat_mpi.c` → domain decomposition, halo exchange, point-to-point MPI
- `kmeans_serial.c` → serial baseline for data-parallel clustering
- `kmeans_mpi.c` → collectives, reductions, broadcasts, convergence detection
- `A2-WRITEUP.txt` → interpreting performance and communication overhead
- assignment HTML page → problem requirements and intended design choices

---

## 8. If you get a compare/contrast exam question

A strong answer would say:

- **Heat** is a nearest-neighbor stencil-style problem, so it is parallelized by spatial decomposition with halo exchange.
- **K-means** is a data-parallel reduction-style problem, so it is parallelized by distributing points and using collectives to combine and redistribute global state.
- **Dense matrix multiplication** often uses row/block decomposition with broadcasts of shared matrix data and gathers of results.
- **LU factorization** is more stage-dependent, often requiring repeated broadcasts of pivot information and synchronized local block updates.

That answer shows you understand not just the code, but the communication structure behind each algorithm.
