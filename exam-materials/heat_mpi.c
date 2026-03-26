#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/*
 * heat_mpi.c
 * ----------
 * Study-note version of the MPI heat solver.
 *
 * Big idea:
 * - The global 1D rod is split by columns across processes.
 * - Each process stores only its own contiguous chunk of columns.
 * - To update boundary cells of that chunk, neighboring processes exchange
 *   one "ghost" value on the left and right every timestep.
 * - At the end, process 0 gathers all local histories and prints the full matrix.
 *
 * Midterm connections:
 * - MPI basics: rank, size, communicator, initialization/finalization.
 * - Point-to-point communication: MPI_Sendrecv for neighbor exchange.
 * - Collective communication: MPI_Gather to reassemble distributed results.
 * - Parallelization strategy: domain decomposition / input partitioning.
 */

/*
 * print_results()
 * ---------------
 * Process 0 uses this helper after gathering the full matrix.
 * H is stored as a flattened 2D array in row-major layout:
 *   H[t * width + p]
 * where t = time index and p = rod position.
 */
static void print_results(double *H, int max_time, int width) {
  int t, p;

  /* Print column headers: rod positions 0..width-1 */
  printf("%3s| ", "");
  for (p = 0; p < width; p++) {
    printf("%5d ", p);
  }
  printf("\n");

  /* Print separator line */
  printf("%3s+-", "---");
  for (p = 0; p < width; p++) {
    printf("------");
  }
  printf("\n");

  /* Print each timestep row */
  for (t = 0; t < max_time; t++) {
    printf("%3d| ", t);
    for (p = 0; p < width; p++) {
      printf("%5.1f ", H[t * width + p]);
    }
    printf("\n");
  }
}

int main(int argc, char **argv) {
  int rank, nprocs;

  /*
   * Every MPI program begins with MPI_Init and usually ends with MPI_Finalize.
   * MPI_COMM_WORLD is the default communicator containing all launched processes.
   */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);   /* rank = this process ID */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* nprocs = total process count */

  /*
   * Keep usage printing on rank 0 only so we do not spam the terminal.
   * This is a common MPI convention: root handles most user-facing I/O.
   */
  if (argc < 4) {
    if (rank == 0) {
      printf("usage: %s max_time width print\n", argv[0]);
      printf("  max_time: int\n");
      printf("  width: int\n");
      printf("  print: 1 print output, 0 no printing\n");
    }
    MPI_Finalize();
    return 0;
  }

  int max_time = atoi(argv[1]);
  int width = atoi(argv[2]);
  int print = atoi(argv[3]);

  /* Physical constants copied from the serial version */
  double initial_temp = 50.0;
  double L_bound_temp = 20.0;
  double R_bound_temp = 10.0;
  double k = 0.5;

  /*
   * This implementation assumes a simple even partition:
   *   width must divide evenly among processes.
   * It also assumes at least 3 columns per process, matching assignment rules.
   */
  if (width % nprocs != 0 || width < 3 * nprocs || max_time < 1) {
    if (rank == 0) {
      fprintf(stderr,
              "Unsupported configuration: width must be divisible by number of processes,\n"
              "width must be at least 3 * nprocs, and max_time must be >= 1.\n");
    }
    MPI_Finalize();
    return 0;
  }

  /*
   * local_cols: number of rod positions owned by this process.
   * start_col:  first global column index owned by this process.
   */
  int local_cols = width / nprocs;
  int start_col = rank * local_cols;

  /*
   * Hlocal stores the full time-history for this process's chunk only.
   * Layout is row-major by time, then local column.
   * Size = max_time * local_cols.
   */
  double *Hlocal = malloc(sizeof(double) * max_time * local_cols);

  /*
   * cur and next are the standard two-buffer optimization.
   * We only need the current timestep to compute the next timestep.
   *
   * Extra two slots are ghost cells:
   *   cur[0]             = left neighbor halo
   *   cur[local_cols+1]  = right neighbor halo
   * Real owned cells are cur[1..local_cols].
   */
  double *cur = malloc(sizeof(double) * (local_cols + 2));
  double *next = malloc(sizeof(double) * (local_cols + 2));

  if (Hlocal == NULL || cur == NULL || next == NULL) {
    fprintf(stderr, "Process %d: memory allocation failed\n", rank);
    free(Hlocal);
    free(cur);
    free(next);
    MPI_Finalize();
    return 0;
  }

  /*
   * Initialize local cells based on their global position.
   * Global endpoints are fixed-temperature boundary conditions.
   * All interior cells start at initial_temp.
   */
  int i, t;
  for (i = 1; i <= local_cols; i++) {
    int global_col = start_col + (i - 1);

    if (global_col == 0) {
      cur[i] = L_bound_temp;
    } else if (global_col == width - 1) {
      cur[i] = R_bound_temp;
    } else {
      cur[i] = initial_temp;
    }

    /* Save timestep 0 into the local history matrix */
    Hlocal[i - 1] = cur[i];
  }

  /*
   * Initialize ghost cells.
   * For physical boundaries we use fixed temperatures.
   * For interior process boundaries, values will be overwritten via communication.
   */
  cur[0] = (rank == 0) ? L_bound_temp : 0.0;
  cur[local_cols + 1] = (rank == nprocs - 1) ? R_bound_temp : 0.0;

  /*
   * Time stepping loop.
   * Each iteration has two phases:
   *   1) exchange boundary data with neighbors
   *   2) compute next local state from cur + ghost values
   */
  for (t = 0; t < max_time - 1; t++) {
    /*
     * Exchange with left neighbor.
     * Send my first owned cell left, receive left neighbor's last owned cell.
     * MPI_Sendrecv avoids deadlock risk from naive blocking send/recv ordering.
     */
    if (rank > 0) {
      MPI_Sendrecv(&cur[1], 1, MPI_DOUBLE, rank - 1, 0,
                   &cur[0], 1, MPI_DOUBLE, rank - 1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      cur[0] = L_bound_temp;
    }

    /*
     * Exchange with right neighbor.
     * Send my last owned cell right, receive right neighbor's first owned cell.
     */
    if (rank < nprocs - 1) {
      MPI_Sendrecv(&cur[local_cols], 1, MPI_DOUBLE, rank + 1, 1,
                   &cur[local_cols + 1], 1, MPI_DOUBLE, rank + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      cur[local_cols + 1] = R_bound_temp;
    }

    /*
     * Update each owned cell.
     * Endpoints remain constant.
     * Interior points use the finite-difference heat equation.
     */
    for (i = 1; i <= local_cols; i++) {
      int global_col = start_col + (i - 1);

      if (global_col == 0) {
        next[i] = L_bound_temp;
      } else if (global_col == width - 1) {
        next[i] = R_bound_temp;
      } else {
        double left_diff = cur[i] - cur[i - 1];
        double right_diff = cur[i] - cur[i + 1];
        double delta = -k * (left_diff + right_diff);
        next[i] = cur[i] + delta;
      }

      /* Save this timestep into the local history matrix */
      Hlocal[(t + 1) * local_cols + (i - 1)] = next[i];
    }

    /* Swap buffers instead of copying arrays */
    double *tmp = cur;
    cur = next;
    next = tmp;
  }

  /*
   * Root-only output path.
   * MPI_Gather collects each process's local history block onto rank 0.
   * Because gathered blocks arrive process-by-process, root then reshapes them
   * into full rows for readable printing.
   */
  if (print) {
    double *Hall = NULL;

    if (rank == 0) {
      Hall = malloc(sizeof(double) * max_time * width);
      if (Hall == NULL) {
        fprintf(stderr, "Process 0: memory allocation failed\n");
        free(Hlocal);
        free(cur);
        free(next);
        MPI_Finalize();
        return 0;
      }
    }

    MPI_Gather(Hlocal, max_time * local_cols, MPI_DOUBLE,
               Hall,   max_time * local_cols, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
      /*
       * Hall layout after gather:
       *   [all times for rank0][all times for rank1]...[all times for rankN-1]
       *
       * But print_results wants full global rows by time, so reorder into Hprint.
       */
      double *Hprint = malloc(sizeof(double) * max_time * width);
      if (Hprint == NULL) {
        fprintf(stderr, "Process 0: memory allocation failed\n");
        free(Hall);
        free(Hlocal);
        free(cur);
        free(next);
        MPI_Finalize();
        return 0;
      }

      int r;
      for (r = 0; r < nprocs; r++) {
        for (t = 0; t < max_time; t++) {
          memcpy(&Hprint[t * width + r * local_cols],
                 &Hall[r * (max_time * local_cols) + t * local_cols],
                 sizeof(double) * local_cols);
        }
      }

      print_results(Hprint, max_time, width);

      free(Hprint);
      free(Hall);
    }
  }

  free(Hlocal);
  free(cur);
  free(next);

  MPI_Finalize();
  return 0;
}
