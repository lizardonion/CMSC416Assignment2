# CMSC416 A2 Midterm Review Sheet

This is the fast-review version of the study material.

Use it for:
- likely short-answer questions
- compare/contrast questions
- memorizing which MPI operation fits which pattern
- connecting A2 to dense matrix multiplication and LU factorization

---

# 1. One-page cram summary

## MPI basics

- **MPI_Init / MPI_Finalize**: start and stop MPI runtime
- **MPI_COMM_WORLD**: communicator containing all launched processes
- **rank**: process ID within communicator
- **size / nprocs**: total number of processes
- **root**: usually rank 0, often handles I/O and orchestration
- **SPMD**: Single Program, Multiple Data
  - every process runs the same code
  - behavior changes based on rank

## Collective ops to memorize

- **MPI_Bcast** = one process sends same data to everybody
- **MPI_Scatter** = root splits equal chunks to everybody
- **MPI_Scatterv** = root splits variable-sized chunks to everybody
- **MPI_Gather** = root collects equal chunks from everybody
- **MPI_Gatherv** = root collects variable-sized chunks from everybody
- **MPI_Reduce** = combine many local values into one result at root
- **MPI_Allreduce** = combine many local values and give final result to everybody

## A2 Problem 1: heat_mpi

- parallelization type: **domain decomposition / spatial decomposition**
- split rod columns among processes
- each process owns a contiguous chunk
- neighbor values needed at chunk boundaries
- exchange halo / ghost-cell values every timestep
- use `MPI_Sendrecv` for left/right neighbor communication
- use `MPI_Gather` to collect final local histories on root

## A2 Problem 3: kmeans_mpi

- parallelization type: **data parallelism over points**
- root reads full dataset
- root distributes points with `MPI_Scatterv`
- each rank computes partial sums for cluster centroids
- root combines partial sums with `MPI_Reduce`
- root broadcasts new centroids with `MPI_Bcast`
- each rank reassigns its local points
- convergence checked with `MPI_Allreduce`
- final assignments collected with `MPI_Gatherv`

## Big comparison

- **Heat**: local dependency, nearest-neighbor communication, frequent small communication
- **K-means**: independent local work per iteration, collective communication at phase boundaries
- **Dense MM**: often row/block decomposition with broadcasts and gathers
- **LU**: stage-based dependency, repeated broadcasts/pivot sharing, synchronized updates

---

# 2. Likely exam questions with model answers

## Q1. What is the purpose of `MPI_Comm_rank()` and `MPI_Comm_size()`?

**Model answer:**
`MPI_Comm_rank()` returns the calling process's rank within a communicator, and `MPI_Comm_size()` returns the total number of processes in that communicator. In MPI programs, ranks are used to assign roles such as root vs non-root and to determine which part of the data each process owns.

---

## Q2. What does SPMD mean, and how does A2 use it?

**Model answer:**
SPMD means Single Program, Multiple Data. All MPI processes run the same executable, but each behaves differently depending on its rank and local data. In A2, all ranks run the same heat or K-means program, but rank 0 usually performs I/O while each rank computes on a different subset of the data.

---

## Q3. Why is rank 0 often used as the root process?

**Model answer:**
Using rank 0 as root is a standard MPI convention. It simplifies program structure by centralizing tasks like reading input files, printing results, gathering final outputs, and coordinating communication.

---

## Q4. How is the heat problem parallelized in `heat_mpi.c`?

**Model answer:**
The rod is divided into contiguous column chunks, and each process owns one chunk. Since each cell update depends on its left and right neighbors, processes exchange boundary values with adjacent ranks every timestep using neighbor communication. After all timesteps are computed, root gathers the distributed results and prints the full matrix.

---

## Q5. Why are ghost cells needed in the heat MPI solution?

**Model answer:**
Ghost cells store neighboring boundary values received from adjacent processes. They allow a process to update its edge-owned cells using the same local formula as interior cells, even though those neighbor values belong to another process.

---

## Q6. Why is `MPI_Sendrecv` a good choice in the heat solver?

**Model answer:**
`MPI_Sendrecv` allows a process to send data and receive data in one coordinated operation. It is especially useful for neighbor exchanges because it avoids deadlock problems that can happen if processes use blocking sends and receives in inconsistent orders.

---

## Q7. How is K-means parallelized in `kmeans_mpi.c`?

**Model answer:**
The dataset points are partitioned across processes. Root reads the full input and distributes subsets of points using `MPI_Scatterv`. Each process computes partial cluster sums and reassigns its local points. The partial sums are combined on root using `MPI_Reduce`, root computes updated centroids, and then broadcasts them with `MPI_Bcast`. Convergence is checked globally with `MPI_Allreduce`, and final assignments are gathered with `MPI_Gatherv`.

---

## Q8. Why does `kmeans_mpi.c` use `MPI_Scatterv` instead of `MPI_Scatter`?

**Model answer:**
`MPI_Scatterv` is used because the number of data points may not divide evenly by the number of processes. The variable-size version lets root send different numbers of points to different processes.

---

## Q9. Why is `MPI_Reduce` used for centroid sums in K-means?

**Model answer:**
Each process computes local partial sums for each cluster. Root needs the global sums in order to compute the new centroids, so `MPI_Reduce` combines all local partial sums into one result on root.

---

## Q10. Why is `MPI_Allreduce` used for the convergence test in K-means?

**Model answer:**
Each process knows only how many of its own local assignments changed. The total number of changes across all processes determines whether the algorithm has converged. Since every process needs that same global value to decide whether to continue iterating, `MPI_Allreduce` is used instead of `MPI_Reduce`.

---

## Q11. What is the difference between `MPI_Reduce` and `MPI_Allreduce`?

**Model answer:**
`MPI_Reduce` combines local values into one result that is delivered only to the root process. `MPI_Allreduce` also combines local values, but the final result is delivered to every process.

---

## Q12. What is the difference between `MPI_Gather` and `MPI_Gatherv`?

**Model answer:**
`MPI_Gather` assumes every process sends the same amount of data to the root. `MPI_Gatherv` allows different processes to send different amounts of data, with root specifying counts and displacements.

---

## Q13. Why is the heat solver less likely to scale well than K-means?

**Model answer:**
The heat solver requires communication with neighboring processes every timestep, so communication overhead can dominate when each process has too little local work. K-means typically has more local computation per communication phase, since each process can do substantial work on its points before synchronizing.

---

## Q14. What kind of parallelization does heat use versus K-means?

**Model answer:**
Heat uses spatial or domain decomposition, where the rod is partitioned by position. K-means uses data parallelism over input points, where points are partitioned across processes.

---

## Q15. What is load balance, and how does A2 handle it?

**Model answer:**
Load balance means dividing work so processes have roughly equal amounts of computation. In heat, the rod is evenly split by columns, so balance is straightforward. In K-means, points are distributed approximately evenly using count/displacement arrays so each process receives nearly the same number of points.

---

# 3. Compare/contrast questions

## Compare heat and K-means in A2

**Strong answer:**
The heat problem has local nearest-neighbor dependencies, so it is parallelized by splitting the rod into contiguous chunks and exchanging halo values between adjacent processes every timestep. K-means has independent per-point work within each iteration, so it is parallelized by distributing points across processes, computing local partial results, and combining them through collectives such as `Reduce`, `Bcast`, and `Allreduce`.

## Compare point-to-point and collective communication in A2

**Strong answer:**
Point-to-point communication appears in the heat solver because only adjacent processes need to exchange boundary values. Collective communication dominates in K-means because all processes repeatedly need shared metadata, global sums, global convergence information, and final output assembly.

## Compare `Reduce` and `Gather`

**Strong answer:**
`Reduce` combines many local values using an operation like sum, max, or min, producing one aggregated result. `Gather` does not combine values; it simply collects data pieces from all processes onto root.

---

# 4. How to recognize which MPI operation to use

## If one process has data that everyone needs
Use **`MPI_Bcast`**

A2 example:
- root broadcasts centroids in K-means

## If root must distribute chunks of an input array
Use **`MPI_Scatter`** or **`MPI_Scatterv`**

A2 example:
- root distributes feature vectors in K-means

## If all processes compute partial sums and only root needs the result
Use **`MPI_Reduce`**

A2 example:
- centroid partial sums and cluster counts

## If all processes compute partial values and everybody needs the final result
Use **`MPI_Allreduce`**

A2 example:
- global count of assignment changes

## If root must collect raw pieces from all processes
Use **`MPI_Gather`** or **`MPI_Gatherv`**

A2 examples:
- gather heat histories
- gather K-means final assignments

## If only neighboring ranks exchange edge data
Use point-to-point ops like **`MPI_Sendrecv`**

A2 example:
- heat halo exchange

---

# 5. Dense matrix multiplication connections

## Likely exam prompt
“How do the ideas from A2 transfer to dense matrix multiplication?”

**Model answer:**
A2 teaches how to choose a decomposition based on data dependencies. In dense matrix multiplication, processes may be assigned rows or blocks of the matrices. One common approach is to scatter rows of `A`, broadcast all of `B`, compute local rows of `C`, and gather the result. This is similar to K-means, where root distributes input data, processes compute local work, and then results are combined or collected.

## Key points to mention

- row-wise or block-wise decomposition
- need to know which matrix data is local versus shared
- `Bcast` may be used for shared matrix data
- `Gather` or block collection may be used for outputs
- communication/computation tradeoff determines scalability

---

# 6. LU factorization connections

## Likely exam prompt
“How does A2 help you reason about parallel LU factorization?”

**Model answer:**
A2 builds the habit of identifying what data is local, what data must be shared, and when synchronization is required. In parallel LU factorization, processes usually own rows or blocks of the matrix, but pivot rows, pivot columns, or panels often need to be shared across processes at each stage. That leads to repeated broadcast-and-update patterns, similar in spirit to the broadcast of centroids in K-means, but with stronger stage-by-stage dependencies.

## Key points to mention

- LU is more sequential across pivot stages than K-means
- processes still do local updates on owned data
- pivot or panel data often must be broadcast
- synchronization is needed between stages
- decomposition quality strongly affects performance

---

# 7. Fast “why” answers

## Why use collectives instead of many manual sends/receives?
Because collectives match common communication patterns directly, simplify code, reduce error-prone coordination logic, and are often optimized by the MPI implementation.

## Why can communication kill speedup?
Because adding more processes can reduce computation per process while communication and synchronization costs remain or increase.

## Why does problem structure matter more than just adding processors?
Because good parallel performance depends on data dependencies, communication volume, load balance, and synchronization frequency.

## Why is K-means often more parallel-friendly than stencil-style heat updates?
Because K-means has more independent local work per synchronization step, while stencil-style heat updates require frequent communication between neighboring partitions.

---

# 8. Very short bullet answers to memorize

## Heat in one line
Split the rod by columns, exchange boundary values with neighbors every timestep, then gather results on root.

## K-means in one line
Scatter points, reduce partial centroid sums, broadcast new centroids, allreduce changes, then gather assignments.

## `Reduce` in one line
Combine local values to root.

## `Allreduce` in one line
Combine local values and return result to everyone.

## `Scatterv` in one line
Distribute variable-sized chunks from root.

## `Gatherv` in one line
Collect variable-sized chunks on root.

## Root in one line
Rank 0 usually handles I/O and coordination.

---

# 9. If you only have 60 seconds before the exam

Memorize this:

- MPI = distributed memory, explicit communication
- rank = process ID, size = total processes, root = rank 0
- Heat = domain decomposition + halo exchange + `MPI_Sendrecv`
- K-means = point partitioning + `Scatterv` + `Reduce` + `Bcast` + `Allreduce` + `Gatherv`
- `Reduce` = root only gets answer
- `Allreduce` = everyone gets answer
- Matrix multiply = distribute rows/blocks, broadcast shared data, gather results
- LU = repeated stage-based broadcasts and local updates
