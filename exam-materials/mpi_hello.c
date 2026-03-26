/*
 * mpi_hello.c
 * -----------
 * Small MPI demo used by the assignment as a warmup.
 *
 * Study value:
 * - Shows the absolute basics of MPI process startup.
 * - Demonstrates rank/size queries.
 * - Demonstrates the common "root-only" output pattern.
 * - Includes a debug print helper that tags messages with process info.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <errno.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <sys/types.h>

/*
 * dpprintf()
 * ----------
 * Debug print helper.
 * If the DEBUG environment variable is set, print a message prefixed with:
 * - process/rank number
 * - total process count
 * - host name
 * - OS process ID
 *
 * This is useful when debugging distributed runs where output from many ranks
 * gets interleaved.
 */
void dpprintf(const char* format, ...) {
  if(getenv("DEBUG") != NULL){
    int total_procs, proc_id, host_len;
    char host[256];

    MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);     /* current process rank */
    MPI_Comm_size (MPI_COMM_WORLD, &total_procs); /* total ranks in communicator */
    MPI_Get_processor_name(host, &host_len);      /* machine name */
    pid_t pid = getpid();

    va_list args;
    va_start (args, format);
    char fmt_buf[2048];
    snprintf(fmt_buf, 2048, "|DEBUG Proc %03d / %d PID %d Host %s| %s",
             proc_id, total_procs, pid, host, format);
    vfprintf(stderr, fmt_buf, args);
    va_end(args);
  }
}

int main (int argc, char *argv[]){
  int proc_total, proc_id;

  /* Standard MPI program startup */
  MPI_Init (&argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size (MPI_COMM_WORLD, &proc_total);

  /* Every rank executes the same program text, but sees different proc_id values. */
  printf("Hello world from process %d of %d\n",
         proc_id, proc_total);

  /* Optional debug output */
  dpprintf("Debug message from processor %d\n",proc_id);

  /* A very common MPI convention: only root handles some user-visible output. */
  if(proc_id == 0){
    printf("Hello from the ROOT processor %d\n", proc_id);
  }

  MPI_Finalize();
  return 0;
}
