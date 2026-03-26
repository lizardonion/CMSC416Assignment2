#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>

/*
 * kmeans_mpi.c
 * ------------
 * Study-note version of the MPI K-means solver.
 *
 * High-level algorithm:
 * 1) Root loads the full dataset.
 * 2) Root scatters subsets of data points to all processes.
 * 3) Each process computes partial sums for the clusters represented in its local data.
 * 4) Root reduces those partial sums into global sums and forms new centroids.
 * 5) Root broadcasts updated centroids back to everyone.
 * 6) Every process reassigns its local points and reports local changes/counts.
 * 7) Repeat until convergence or max iterations.
 * 8) Root gathers final assignments and writes outputs.
 *
 * Midterm connections:
 * - MPI basics: ranks, root process, communicators, launch model.
 * - Collectives: Bcast, Scatterv, Reduce, Allreduce, Gatherv.
 * - Effective parallelization: data parallelism over points, minimize I/O to root.
 * - Pattern recognition: this is a classic map/reduce + broadcast iteration.
 */

int filestats(char *filename, ssize_t *tot_tokens, ssize_t *tot_lines);

typedef struct {
    int ndata;      /* total number of points in the full dataset */
    int dim;        /* features per point */
    float *features;/* flattened feature matrix: point i starts at i*dim */
    int *assigns;   /* cluster assignment for each point */
    int *labels;    /* true digit labels, used for confusion matrix only */
    int nlabels;    /* number of distinct labels */
} KMData;

typedef struct {
    int nclust;      /* number of clusters */
    int dim;         /* features per centroid */
    float *features; /* flattened centroid matrix */
    int *counts;     /* number of points currently assigned to each cluster */
} KMClust;

/*
 * kmdata_load()
 * -------------
 * Root-only helper that reads the text dataset into contiguous arrays.
 * Each input line contains:
 *   label : feature_0 feature_1 ... feature_(dim-1)
 */
KMData kmdata_load(const char *datafile) {
    KMData data;
    memset(&data, 0, sizeof(KMData));

    ssize_t tot_tokens = 0, tot_lines = 0;
    if (filestats((char *)datafile, &tot_tokens, &tot_lines) != 0) {
        printf("Failed to open file '%s'\n", datafile);
        exit(1);
    }

    int tokens_per_line = (int)(tot_tokens / tot_lines);
    data.ndata = (int)tot_lines;
    data.dim = tokens_per_line - 2; /* subtract label and colon token */

    data.features = (float *)malloc((size_t)data.ndata * (size_t)data.dim * sizeof(float));
    data.assigns = (int *)malloc((size_t)data.ndata * sizeof(int));
    data.labels = (int *)malloc((size_t)data.ndata * sizeof(int));

    if (!data.features || !data.assigns || !data.labels) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    FILE *fin = fopen(datafile, "r");
    if (!fin) {
        printf("Failed to open file '%s'\n", datafile);
        exit(1);
    }

    int max_label = -1;

    for (int i = 0; i < data.ndata; i++) {
        int label;
        char colon[8];

        fscanf(fin, "%d %7s", &label, colon);
        data.labels[i] = label;
        if (label > max_label) max_label = label;

        for (int d = 0; d < data.dim; d++) {
            float val;
            fscanf(fin, "%f", &val);
            data.features[(size_t)i * (size_t)data.dim + (size_t)d] = val;
        }
    }

    fclose(fin);

    data.nlabels = max_label + 1;
    return data;
}

/* Create a zero-initialized cluster structure. */
KMClust kmclust_new(int nclust, int dim) {
    KMClust clust;
    clust.nclust = nclust;
    clust.dim = dim;

    clust.features = (float *)malloc((size_t)nclust * (size_t)dim * sizeof(float));
    clust.counts = (int *)malloc((size_t)nclust * sizeof(int));

    if (!clust.features || !clust.counts) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    for (int i = 0; i < nclust * dim; i++) clust.features[i] = 0.0f;
    for (int i = 0; i < nclust; i++) clust.counts[i] = 0;

    return clust;
}

/*
 * save_pgm_files()
 * ----------------
 * Root-only visualization helper.
 * If the feature dimension is a square (like 28x28 MNIST pixels), write each
 * centroid back out as a grayscale PGM image.
 */
void save_pgm_files(KMClust *clust, const char *savedir) {
    int dim_root = (int)(sqrt((double)clust->dim));
    if (dim_root * dim_root != clust->dim) return;

    float maxfeat = 0.0f;
    for (int i = 0; i < clust->nclust * clust->dim; i++) {
        if (clust->features[i] > maxfeat) maxfeat = clust->features[i];
    }
    if (maxfeat < 1.0f) maxfeat = 1.0f;

    printf("Saving cluster centers to %s/cent_0000.pgm ...\n", savedir);

    for (int c = 0; c < clust->nclust; c++) {
        char outfile[512];
        sprintf(outfile, "%s/cent_%04d.pgm", savedir, c);

        FILE *pgm = fopen(outfile, "w");
        if (!pgm) {
            printf("Failed to open file '%s'\n", outfile);
            exit(1);
        }

        fprintf(pgm, "P2\n");
        fprintf(pgm, "%d %d\n", dim_root, dim_root);
        fprintf(pgm, "%.0f\n", maxfeat);

        for (int d = 0; d < clust->dim; d++) {
            if (d > 0 && d % dim_root == 0) fprintf(pgm, "\n");
            fprintf(pgm, "%3.0f ", clust->features[c * clust->dim + d]);
        }
        fprintf(pgm, "\n");
        fclose(pgm);
    }
}

/*
 * build_counts_displs()
 * ---------------------
 * Precompute the arguments needed for variable-size scatter/gather operations.
 *
 * Why Scatterv/Gatherv instead of Scatter/Gather?
 * - ndata may not be evenly divisible by nprocs.
 * - So some ranks get one extra point.
 * - The *v variants let root specify per-rank counts and displacements.
 */
static void build_counts_displs(int ndata, int nprocs, int dim,
                                int *point_counts, int *point_displs,
                                int *feat_counts, int *feat_displs) {
    int base = ndata / nprocs;
    int extra = ndata % nprocs;

    int pdisp = 0;
    int fdisp = 0;
    for (int p = 0; p < nprocs; p++) {
        int local_n = base + (p < extra ? 1 : 0);
        point_counts[p] = local_n;
        point_displs[p] = pdisp;
        feat_counts[p] = local_n * dim;
        feat_displs[p] = fdisp;
        pdisp += local_n;
        fdisp += local_n * dim;
    }
}

int main(int argc, char **argv) {
    /*
     * Standard MPI startup.
     * This program uses MPI_COMM_WORLD only.
     */
    MPI_Init(&argc, &argv);

    /* Avoid Open MPI / Valgrind startup issues from CUDA-aware runtime setup. */
    setenv("OMPI_MCA_opal_cuda_support", "0", 1);

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /*
     * Root-only usage printing, another common MPI convention.
     */
    if (argc < 3) {
        if (rank == 0) {
            printf("usage: kmeans_serial <datafile> <nclust> [savedir] [maxiter]\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *datafile = argv[1];
    int nclust = atoi(argv[2]);
    const char *savedir = ".";
    int MAXITER = 100;

    if (argc > 3) savedir = argv[3];
    if (argc > 4) MAXITER = atoi(argv[4]);

    KMData data;
    memset(&data, 0, sizeof(KMData));
    KMClust clust;
    memset(&clust, 0, sizeof(KMClust));

    /*
     * Only rank 0 performs file I/O, exactly as the assignment requires.
     * Other processes will receive the metadata and data they need via MPI.
     */
    if (rank == 0) {
        char cmd[512];
        sprintf(cmd, "mkdir -p %s", savedir);
        system(cmd);

        printf("datafile: %s\n", datafile);
        printf("nclust: %d\n", nclust);
        printf("savedir: %s\n", savedir);

        data = kmdata_load(datafile);
        clust = kmclust_new(nclust, data.dim);

        printf("ndata: %d\n", data.ndata);
        printf("dim: %d\n\n", data.dim);
    }

    /*
     * Broadcast scalar metadata so every process knows the problem size.
     * This is a classic use of MPI_Bcast.
     */
    MPI_Bcast(&nclust, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&MAXITER, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.ndata, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&data.dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * Non-root ranks can now allocate their cluster arrays because they know dim.
     */
    if (rank != 0) {
        clust = kmclust_new(nclust, data.dim);
    }
    clust.nclust = nclust;
    clust.dim = data.dim;

    /*
     * Build partition metadata for points and flattened feature arrays.
     */
    int *point_counts = (int *)malloc((size_t)nprocs * sizeof(int));
    int *point_displs = (int *)malloc((size_t)nprocs * sizeof(int));
    int *feat_counts = (int *)malloc((size_t)nprocs * sizeof(int));
    int *feat_displs = (int *)malloc((size_t)nprocs * sizeof(int));
    if (!point_counts || !point_displs || !feat_counts || !feat_displs) {
        printf("Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    build_counts_displs(data.ndata, nprocs, data.dim,
                        point_counts, point_displs,
                        feat_counts, feat_displs);

    int local_ndata = point_counts[rank];
    int global_start = point_displs[rank];

    /*
     * Local storage:
     * - local_features: this rank's subset of the points
     * - local_assigns: current local cluster assignments
     * - local_sum_features: partial sums for centroid recomputation
     * - local_new_counts: local membership counts after reassignment
     */
    float *local_features = (float *)malloc((size_t)local_ndata * (size_t)data.dim * sizeof(float));
    int *local_assigns = (int *)malloc((size_t)local_ndata * sizeof(int));
    float *local_sum_features = (float *)malloc((size_t)nclust * (size_t)data.dim * sizeof(float));
    int *local_new_counts = (int *)malloc((size_t)nclust * sizeof(int));
    float *global_sum_features = NULL;

    if (!local_features || !local_assigns || !local_sum_features || !local_new_counts) {
        printf("Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Root needs an extra receive buffer for the Reduce of centroid sums. */
    if (rank == 0) {
        global_sum_features = (float *)malloc((size_t)nclust * (size_t)data.dim * sizeof(float));
        if (!global_sum_features) {
            printf("Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /*
     * Scatter the actual point data.
     * This is the key data-distribution step of the parallelization.
     */
    MPI_Scatterv(data.features, feat_counts, feat_displs, MPI_FLOAT,
                 local_features, feat_counts[rank], MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    /*
     * Initial assignments match the serial implementation:
     * point i starts in cluster i % nclust.
     * Each rank computes its local slice consistently using the global start index.
     */
    for (int i = 0; i < local_ndata; i++) {
        local_assigns[i] = (global_start + i) % nclust;
    }

    /* Root initializes global assignment/count arrays for reporting and first centroid update. */
    if (rank == 0) {
        for (int i = 0; i < data.ndata; i++) {
            data.assigns[i] = i % nclust;
        }
        for (int c = 0; c < nclust; c++) {
            int icount = data.ndata / nclust;
            int extra = (c < (data.ndata % nclust)) ? 1 : 0;
            clust.counts[c] = icount + extra;
        }

        printf("==CLUSTERING: MAXITER %d==\n", MAXITER);
        printf("ITER NCHANGE CLUST_COUNTS\n");
    }

    int curiter = 1;
    int nchanges = data.ndata;

    /*
     * Main iterative K-means loop.
     *
     * Observe the communication pattern:
     *   local work  -> Reduce    -> root computes centroids
     *   root state  -> Bcast     -> all ranks reassign
     *   local stats -> Reduce/Allreduce -> convergence test + counts
     */
    while (nchanges > 0 && curiter <= MAXITER) {
        /* Zero local partial sums before recomputing centroids. */
        for (int i = 0; i < nclust * clust.dim; i++) {
            local_sum_features[i] = 0.0f;
        }

        /*
         * Accumulate local feature sums by current local assignments.
         * This is the "map" part of the iteration.
         */
        for (int i = 0; i < local_ndata; i++) {
            int c = local_assigns[i];
            for (int d = 0; d < clust.dim; d++) {
                local_sum_features[(size_t)c * (size_t)clust.dim + (size_t)d] +=
                    local_features[(size_t)i * (size_t)data.dim + (size_t)d];
            }
        }

        /*
         * Sum all local partial centroid sums onto root.
         * MPI_Reduce is enough here because only root needs the combined result.
         */
        MPI_Reduce(local_sum_features, global_sum_features, nclust * clust.dim,
                   MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        /*
         * Root computes actual centroids by dividing summed coordinates by the
         * previous iteration's cluster counts.
         */
        if (rank == 0) {
            for (int i = 0; i < nclust * clust.dim; i++) {
                clust.features[i] = global_sum_features[i];
            }

            for (int c = 0; c < nclust; c++) {
                if (clust.counts[c] > 0) {
                    for (int d = 0; d < clust.dim; d++) {
                        clust.features[(size_t)c * (size_t)clust.dim + (size_t)d] /= (float)clust.counts[c];
                    }
                }
            }
        }

        /*
         * Broadcast new centroids so every process can reassign its local points.
         */
        MPI_Bcast(clust.features, nclust * clust.dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

        for (int c = 0; c < nclust; c++) local_new_counts[c] = 0;
        int local_nchanges = 0;

        /*
         * Reassign each local point to the nearest centroid.
         * Distance metric: squared Euclidean distance.
         */
        for (int i = 0; i < local_ndata; i++) {
            int best_clust = 0;
            float best_distsq = INFINITY;

            for (int c = 0; c < nclust; c++) {
                float distsq = 0.0f;

                for (int d = 0; d < clust.dim; d++) {
                    float diff =
                        local_features[(size_t)i * (size_t)data.dim + (size_t)d] -
                        clust.features[(size_t)c * (size_t)clust.dim + (size_t)d];
                    distsq += diff * diff;
                }

                if (distsq < best_distsq) {
                    best_distsq = distsq;
                    best_clust = c;
                }
            }

            local_new_counts[best_clust]++;

            if (best_clust != local_assigns[i]) {
                local_nchanges++;
                local_assigns[i] = best_clust;
            }
        }

        /*
         * Aggregate new cluster counts onto root for the next iteration.
         * Aggregate total assignment changes onto all ranks for convergence.
         *
         * Why Allreduce for nchanges?
         * - Every process needs the same loop condition.
         * - So the result must be distributed back to all of them.
         */
        MPI_Reduce(local_new_counts, clust.counts, nclust, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Allreduce(&local_nchanges, &nchanges, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("%3d: %5d |", curiter, nchanges);
            for (int c = 0; c < nclust; c++) {
                printf(" %4d", clust.counts[c]);
            }
            printf("\n");
        }

        curiter++;
    }

    /*
     * Gather final assignments back onto root so root can write labels.txt
     * and compute the confusion matrix in serial.
     */
    int *recv_assign_counts = point_counts;
    int *recv_assign_displs = point_displs;

    MPI_Gatherv(local_assigns, local_ndata, MPI_INT,
                data.assigns, recv_assign_counts, recv_assign_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (curiter > MAXITER) {
        if (rank == 0) printf("WARNING: maximum iteration %d exceeded\n", MAXITER);
    }
    else {
        if (rank == 0) printf("CONVERGED: after %d iterations\n", curiter);
    }

    /*
     * Root-only postprocessing and output.
     * Keeping I/O centralized simplifies correctness and matches assignment rules.
     */
    if (rank == 0) {
        int *confusion = (int *)calloc((size_t)data.nlabels * (size_t)nclust, sizeof(int));
        if (!confusion) {
            printf("Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (int i = 0; i < data.ndata; i++) {
            confusion[(size_t)data.labels[i] * (size_t)nclust +
                      (size_t)data.assigns[i]]++;
        }

        printf("\n==CONFUSION MATRIX + COUNTS==\n");
        printf("LABEL \\ CLUST\n");

        printf("   ");
        for (int c = 0; c < nclust; c++) printf(" %4d", c);
        printf("  TOT\n");

        for (int l = 0; l < data.nlabels; l++) {
            int tot = 0;
            printf("%2d:", l);
            for (int c = 0; c < nclust; c++) {
                int v = confusion[(size_t)l * (size_t)nclust + (size_t)c];
                printf(" %4d", v);
                tot += v;
            }
            printf(" %4d\n", tot);
        }

        printf("TOT");
        int grand = 0;
        for (int c = 0; c < nclust; c++) {
            printf(" %4d", clust.counts[c]);
            grand += clust.counts[c];
        }
        printf(" %4d\n", grand);

        char outfile[512];
        sprintf(outfile, "%s/labels.txt", savedir);
        printf("\nSaving cluster labels to file %s\n", outfile);

        save_pgm_files(&clust, savedir);

        FILE *fout = fopen(outfile, "w");
        if (!fout) {
            printf("Failed to open file '%s'\n", outfile);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < data.ndata; i++) {
            fprintf(fout, "%2d %2d\n", data.labels[i], data.assigns[i]);
        }
        fclose(fout);

        free(confusion);
        free(global_sum_features);
        free(data.features);
        free(data.assigns);
        free(data.labels);
    }

    free(local_features);
    free(local_assigns);
    free(local_sum_features);
    free(local_new_counts);
    free(point_counts);
    free(point_displs);
    free(feat_counts);
    free(feat_displs);
    free(clust.features);
    free(clust.counts);

    MPI_Finalize();
    return 0;
}
