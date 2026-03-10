#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

static void print_results(double *H, int max_time, int width) {
  int t, p;

  printf("%3s| ", "");
  for (p = 0; p < width; p++) {
    printf("%5d ", p);
  }
  printf("\n");

  printf("%3s+-", "---");
  for (p = 0; p < width; p++) {
    printf("------");
  }
  printf("\n");

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

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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

  double initial_temp = 50.0;
  double L_bound_temp = 20.0;
  double R_bound_temp = 10.0;
  double k = 0.5;

  if (width % nprocs != 0 || width < 3 * nprocs || max_time < 1) {
    if (rank == 0) {
      fprintf(stderr,
              "Unsupported configuration: width must be divisible by number of processes,\n"
              "width must be at least 3 * nprocs, and max_time must be >= 1.\n");
    }
    MPI_Finalize();
    return 0;
  }

  int local_cols = width / nprocs;
  int start_col = rank * local_cols;

  double *Hlocal = malloc(sizeof(double) * max_time * local_cols);

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

    Hlocal[i - 1] = cur[i];
  }
  cur[0] = (rank == 0) ? L_bound_temp : 0.0;
  cur[local_cols + 1] = (rank == nprocs - 1) ? R_bound_temp : 0.0;

  for (t = 0; t < max_time - 1; t++) {
    if (rank > 0) {
      MPI_Sendrecv(&cur[1], 1, MPI_DOUBLE, rank - 1, 0,
                   &cur[0], 1, MPI_DOUBLE, rank - 1, 1,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      cur[0] = L_bound_temp;
    }

    if (rank < nprocs - 1) {
      MPI_Sendrecv(&cur[local_cols], 1, MPI_DOUBLE, rank + 1, 1,
                   &cur[local_cols + 1], 1, MPI_DOUBLE, rank + 1, 0,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      cur[local_cols + 1] = R_bound_temp;
    }

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

      Hlocal[(t + 1) * local_cols + (i - 1)] = next[i];
    }
    double *tmp = cur;
    cur = next;
    next = tmp;
  }

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