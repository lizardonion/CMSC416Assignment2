# CMSC416 Assignment 2 Study Guide

This codebase is about **distributed-memory parallel programming with MPI**. The assignment webpage breaks it into three problems:

1. Parallelize the 1D heat simulation with MPI
2. Keep the serial K-means implementation working
3. Parallelize K-means with MPI

The code in this folder now doubles as study material:
- the C files are heavily commented
- this guide explains the big ideas
- the companion notes connect the assignment to likely midterm topics

---

## 1. What the assignment is really teaching

At a high level, A2 is teaching you how to take a serial algorithm and answer four questions:

1. **What data exists globally?**
2. **How should that data be partitioned across processes?**
3. **What information must cross process boundaries?**
4. **Which MPI operations fit that communication pattern?**

That is the real exam skill.

---

## 2. Codebase map

### Core source files

- `mpi_hello.c`
  - small MPI warmup
  - introduces `MPI_Init`, rank, size, root-only printing
- `heat_serial.c`
  - serial reference for Problem 1
- `heat_mpi.c`
  - MPI parallel version of the heat solver
- `kmeans_serial.c`
  - serial K-means reference for Problems 2 and 3
- `kmeans_mpi.c`
  - MPI parallel K-means solver
- `kmeans_util.c`
  - helper used by serial and MPI K-means

### Assignment docs / workflow files

- `CMSC416 Assignment 2_ Distributed Memory Programming with MPI.htm`
  - saved assignment webpage
- `A2-WRITEUP.txt`
  - timing tables and written analysis
- `Makefile`
  - build/test automation
- `heat-slurm.sh`, `kmeans-slurm.sh`
  - batch scripts for Zaratan timing runs

---

## 3. MPI basics and conventions seen in the code

### MPI startup / shutdown

Every MPI program follows this pattern:

```c
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
...
MPI_Finalize();
```

Meaning:
- `MPI_Init` starts the MPI runtime
- `MPI_COMM_WORLD` is the communicator containing all launched processes
- `rank` is the process ID within the communicator
- `nprocs` is the number of total processes

### SPMD model

MPI programs usually follow **SPMD**: **Single Program, Multiple Data**.

That means:
- every process runs the same executable
- each process behaves differently based on `rank`

Example:
- rank 0 may read files and print output
- other ranks compute on their local data only

### Root process convention

This assignment uses **rank 0 as root** for most user-facing work:
- printing usage/help
- reading input files
- creating output directories
- printing final results
- saving output files

That is a common and important MPI convention.

### Distributed-memory mindset

MPI assumes separate process memories.
A process **cannot directly access another process's arrays**.
If it needs another process's data, it must communicate explicitly.

That is the core difference from shared-memory programming.

---

## 4. Problem 1: Heat equation with MPI

## Serial version: `heat_serial.c`

The heat simulation models a 1D rod with:
- fixed left boundary = 20.0
- fixed right boundary = 10.0
- interior initially = 50.0

For each interior position `p` at timestep `t`, the update uses neighboring positions:

```text
left_diff  = H[t][p] - H[t][p-1]
right_diff = H[t][p] - H[t][p+1]
delta      = -k * (left_diff + right_diff)
H[t+1][p]  = H[t][p] + delta
```

### Important dependency fact

Each cell depends only on:
- itself at time `t`
- left neighbor at time `t`
- right neighbor at time `t`

That local dependency pattern is exactly why MPI parallelization is feasible.

---

## MPI version: `heat_mpi.c`

### Parallelization strategy

The rod is partitioned by **columns** across processes.
If width = 12 and processes = 4, then each process owns 3 contiguous columns.

So instead of one process storing all columns, each process stores only its local chunk.

### Local data structures

The MPI version uses:
- `Hlocal`: full time-history for the local chunk only
- `cur`: current timestep values for local chunk plus ghost cells
- `next`: next timestep values for local chunk plus ghost cells

Ghost-cell layout:
- `cur[0]` = left halo value
- `cur[1..local_cols]` = owned cells
- `cur[local_cols+1]` = right halo value

### Why ghost cells?

A process needs neighbor values to update its boundary-owned cells.
Those values belong to adjacent processes.
So before each timestep update, neighbors exchange boundary values.

### Communication used

`heat_mpi.c` uses **point-to-point communication**:
- `MPI_Sendrecv` with left neighbor
- `MPI_Sendrecv` with right neighbor

This is a clean choice because the pattern is strictly nearest-neighbor communication.

### Why `MPI_Sendrecv` matters

If processes used blocking `MPI_Send` and `MPI_Recv` in the wrong order, they could deadlock.
`MPI_Sendrecv` is a standard safe pattern for pairwise exchanges.

### Final collection

After computation, process 0 gathers each process's local history using:
- `MPI_Gather`

Then root rearranges the gathered blocks into full rows for printing.

### Exam takeaway from Problem 1

Problem 1 is a textbook example of:
- **domain decomposition**
- **nearest-neighbor communication**
- **halo exchange / ghost cells**
- **gathering distributed results for output**

---

## 5. Problem 2: Serial K-means

`kmeans_serial.c` is the baseline algorithm that Problem 3 parallelizes.

### K-means phases

The serial algorithm repeatedly does two things:

1. **Recompute centroids**
   - sum points assigned to each cluster
   - divide by cluster count
2. **Reassign points**
   - compute distance from each point to each centroid
   - assign point to closest centroid

Stop when no assignment changes or max iterations is reached.

### Data representation

- `KMData`
  - all points, assignments, labels
- `KMClust`
  - centroids and counts

Features are stored in flattened arrays rather than 2D pointer grids.
That is good for contiguous memory access and MPI communication.

### Why this matters for MPI

The serial code exposes the natural parallel unit:
- each point can be processed independently during centroid accumulation and reassignment

That makes K-means a good fit for **data parallelism over points**.

---

## 6. Problem 3: MPI K-means

`kmeans_mpi.c` parallelizes K-means by partitioning the **input points** across processes.

### Why partition by points?

Because most work is per-point:
- contribute point to cluster sum
- compute point-to-centroid distances
- update assignment

So each process can work mostly independently on its own subset of points.

This is the assignment's intended **input partitioning** strategy.

### High-level iteration in MPI form

Each iteration does this:

1. each process computes local partial sums for centroid coordinates
2. root reduces those into global sums
3. root computes new centroid coordinates
4. root broadcasts centroids to all processes
5. each process reassigns its local points
6. processes combine local change counts into a global count
7. if global changes = 0, stop

### Collective operations used

#### `MPI_Bcast`
Used to broadcast:
- `nclust`
- `MAXITER`
- `ndata`
- `dim`
- updated centroid coordinates

Why: every process needs the same metadata and current centroids.

#### `MPI_Scatterv`
Used to distribute the dataset from root to all processes.

Why `Scatterv` instead of `Scatter`?
- number of points may not divide evenly by number of processes
- some processes may receive one extra point

#### `MPI_Reduce`
Used for:
- summing partial centroid feature sums onto root
- summing local cluster counts onto root

Why `Reduce` instead of `Allreduce` here?
- only root needs the summed result to form new centroids and report counts

#### `MPI_Allreduce`
Used for:
- summing `local_nchanges` into global `nchanges`

Why `Allreduce`?
- every process needs the same convergence result to decide whether to continue looping

#### `MPI_Gatherv`
Used to gather final local assignments back to root.

Why `Gatherv`?
- local point counts may differ by process

### Communication pattern in one sentence

Problem 3 is basically:
- **Scatter once**
- then loop over **Reduce + Bcast + Allreduce**
- then **Gather once**

That is a very exam-friendly way to remember it.

---

## 7. How the assignment demonstrates effective parallelization

A2 is not just about using MPI calls. It is about choosing the right decomposition.

### Heat: why the strategy is reasonable

The heat problem has local dependencies only.
So the natural decomposition is by contiguous rod segments.

Benefits:
- small communication volume
- communication only with neighboring ranks
- simple data ownership

Costs:
- must exchange halo values every timestep
- communication can dominate for small local problem sizes

That lines up with the timing writeup: more processes did not necessarily help.

### K-means: why the strategy is reasonable

K-means has a lot of per-point work.
Each process can independently:
- accumulate local sums
- compute distances for local points
- update local assignments

Communication happens only at synchronization points:
- combine partial sums
- distribute new centroids
- combine convergence counts

That is much more scalable than the heat case when the data is large enough.

### General exam principle

A parallelization is effective when:
- local computation is large
- communication is limited and structured
- load balance is decent
- synchronization is not too frequent

Heat has frequent neighbor communication per timestep.
K-means has heavier local work between collectives.
That helps explain why K-means parallelization tends to look better.

---

## 8. Collective communication: what the code teaches you

This assignment is a nice case study in **when to use which collective**.

### Broadcast
Use when one process has data that everyone needs.

In A2:
- root broadcasts metadata and centroids

### Scatter / Scatterv
Use when one process owns a large input and wants to distribute pieces.

In A2:
- root distributes point subsets for K-means

### Gather / Gatherv
Use when distributed pieces must come back to one process.

In A2:
- root collects final assignments
- root collects heat local histories for printing

### Reduce
Use when many processes contribute partial results but only one process needs the final answer.

In A2:
- root collects partial centroid sums
- root collects cluster counts

### Allreduce
Use when many processes contribute partial results and **all** processes need the final answer.

In A2:
- all ranks need the same global `nchanges` convergence value

---

## 9. Midterm tie-in: dense matrix multiplication parallel algorithms

Dense matrix multiplication is not directly implemented in this assignment, but the same design questions appear.

Suppose `C = A x B`.

### Parallel design questions

You would ask:
- how are rows/columns/blocks partitioned?
- what data does each process need?
- how much data must move?
- when can communication overlap with computation?

### Common parallel decompositions

#### 1D row-wise decomposition
Each process gets a block of rows of `A`.
To compute its rows of `C`, it may need all of `B`.

Typical communication idea:
- scatter rows of `A`
- broadcast all of `B`
- each process computes its rows of `C`
- gather rows of `C`

This should feel familiar because it resembles the K-means pattern:
- distribute independent work
- broadcast shared data
- gather results

#### 2D block decomposition
Each process gets a block of `A` and `B`.
Communication is more complex but often more scalable.

This is conceptually closer to the heat problem's locality ideas, except in 2D.
Processes depend on structured neighbors or stages of data movement.

### What A2 helps you understand for matrix multiply

From A2, you should already recognize:
- partitioning matters more than the raw MPI calls
- communication patterns come from data dependencies
- collectives are often driven by who needs the combined/shared data
- good decomposition minimizes unnecessary communication

---

## 10. Midterm tie-in: LU factorization and parallelization

LU factorization is also not directly implemented here, but A2 gives you the right mental model.

### Serial LU idea

You factor a matrix `A` into:
- `L` = lower triangular
- `U` = upper triangular

The algorithm progresses in stages/pivots.
Each stage updates trailing submatrices based on the current pivot row/column.

### Why LU is harder than K-means

K-means is largely point-independent within an iteration.
LU has stronger sequential dependencies across pivot steps.

That means parallel LU often looks like:
- partition matrix into rows/blocks
- identify pivot row/column work
- broadcast pivot information
- update local trailing blocks in parallel
- synchronize at each major step

### A2 concepts that transfer directly

#### Root/shared-state broadcasts
Like centroids in K-means, pivot data may need to be broadcast.

#### Local partial work + synchronization
Like local K-means point processing or local heat updates, each rank updates its owned data.
Then processes synchronize before the next stage.

#### Structured data partitioning
Like rod segments in heat or point blocks in K-means, LU performance depends heavily on how matrix blocks are assigned.

### Exam contrast worth knowing

- **Heat**: local nearest-neighbor dependency
- **K-means**: embarrassingly data-parallel between synchronizations
- **LU**: stage-based dependency with recurring broadcasts/updates
- **Dense MM**: often regular block-based parallelism with more flexible decomposition choices

That comparison is exactly the kind of thing instructors like to ask.

---

## 11. Fast memorization sheet

### If the exam asks: “What MPI ideas appear in A2?”

Answer with:
- rank / size / communicator basics
- root-process conventions
- point-to-point neighbor exchange
- collectives: `Bcast`, `Scatterv`, `Reduce`, `Allreduce`, `Gather/Gatherv`
- data partitioning / domain decomposition
- ghost cells / halo exchange
- global convergence detection

### If the exam asks: “How was heat parallelized?”

Answer with:
- split rod columns among processes
- each process stores only local chunk
- exchange boundary values with left/right neighbors each timestep
- update local cells
- gather results on root for printing

### If the exam asks: “How was K-means parallelized?”

Answer with:
- root reads full dataset
- scatter points across processes
- each process computes partial cluster sums and local reassignments
- reduce sums/counts to root
- root computes centroids
- broadcast centroids back out
- allreduce assignment changes for convergence
- gatherv final assignments to root

### If the exam asks: “Why does collective choice matter?”

Answer with:
- use `Reduce` when only root needs result
- use `Allreduce` when everybody needs result
- use `Scatterv/Gatherv` when chunks may be uneven
- use `Bcast` when one process has shared state for everyone

---

## 12. Best files to reread before the exam

If short on time, reread in this order:

1. `STUDY_GUIDE.md`
2. `MIDTERM_TOPIC_CONNECTIONS.md`
3. `heat_mpi.c`
4. `kmeans_mpi.c`
5. `heat_serial.c`
6. `kmeans_serial.c`
7. `mpi_hello.c`

That order gives concepts first, then implementation.
