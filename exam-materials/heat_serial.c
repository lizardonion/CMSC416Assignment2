#include <stdio.h>
#include <stdlib.h>

/*
 * heat_serial.c
 * -------------
 * Serial reference implementation for the 1D heat problem.
 *
 * Why this matters for exam study:
 * - It is the baseline algorithm before parallelization.
 * - Understanding the serial data dependencies makes the MPI version much easier.
 * - The MPI version keeps the same numerical update rule and only changes how
 *   the data is distributed and communicated.
 */

/*
 * HEAT TRANSFER MODEL
 * -------------------
 * The rod has fixed temperatures at the left and right endpoints.
 * Interior cells start at the same initial temperature.
 *
 * For each timestep t and interior position p:
 *   left_diff  = H[t][p] - H[t][p-1]
 *   right_diff = H[t][p] - H[t][p+1]
 *   delta      = -k * (left_diff + right_diff)
 *   H[t+1][p]  = H[t][p] + delta
 *
 * Equivalent form:
 *   H[t+1][p] = H[t][p] + k*H[t][p-1] - 2*k*H[t][p] + k*H[t][p+1]
 *
 * Dependency insight:
 * - To compute time t+1, you only need neighboring values from time t.
 * - That local-neighbor dependence is exactly why the MPI version only needs
 *   nearest-neighbor communication.
 */
int main(int argc, char **argv){
  if(argc < 4){
    printf("usage: %s max_time width print\n", argv[0]);
    printf("  max_time: int\n");
    printf("  width: int\n");
    printf("  print: 1 print output, 0 no printing\n");
    return 0;
  }

  int max_time = atoi(argv[1]); /* Number of timesteps */
  int width = atoi(argv[2]);    /* Number of rod positions */
  int print = atoi(argv[3]);

  /* Physical parameters */
  double initial_temp = 50.0;   /* Initial temp for interior cells */
  double L_bound_temp = 20.0;   /* Fixed left endpoint temp */
  double R_bound_temp = 10.0;   /* Fixed right endpoint temp */
  double k = 0.5;               /* Thermal conductivity */
  double **H;                   /* H[t][p] stores temp at time t, position p */

  /*
   * Allocate a full 2D time-by-position matrix.
   * In contrast, the MPI version stores only a local piece of each row.
   */
  H = malloc(sizeof(double*)*max_time);
  int t,p;
  for(t=0;t<max_time;t++){
    H[t] = malloc(sizeof(double)*width);
  }

  /* Boundary columns are fixed for every timestep. */
  for(t=0; t<max_time; t++){
    H[t][0] = L_bound_temp;
    H[t][width-1] = R_bound_temp;
  }

  /* Initialize the first row (time 0). */
  t = 0;
  for(p=1; p<width-1; p++){
    H[t][p] = initial_temp;
  }

  /*
   * Time-marching loop.
   * Each new row depends only on the previous row.
   */
  for(t=0; t<max_time-1; t++){
    for(p=1; p<width-1; p++){
      double left_diff  = H[t][p] - H[t][p-1];
      double right_diff = H[t][p] - H[t][p+1];
      double delta = -k*( left_diff + right_diff );
      H[t+1][p] = H[t][p] + delta;
    }
  }

  if(print){
    /* Print matrix in the exact format expected by tests/assignment. */
    printf("%3s| ","");
    for(p=0; p<width; p++){
      printf("%5d ",p);
    }
    printf("\n");
    printf("%3s+-","---");
    for(p=0; p<width; p++){
      printf("------");
    }
    printf("\n");

    for(t=0; t<max_time; t++){
      printf("%3d| ",t);
      for(p=0; p<width; p++){
        printf("%5.1f ",H[t][p]);
      }
      printf("\n");
    }
  }

  for(t=0; t<max_time; t++){
    free(H[t]);
  }
  free(H);

  return 0;
}
