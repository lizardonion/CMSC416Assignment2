#!/bin/bash -l

# ADJUST: location of project code with respect to home directory
cd YOUR_A2_FOLDER_HERE

## BASIC JOB SETUP
#SBATCH --time=0:14:00
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=250m
#SBATCH --account cmsc416-class

## SELECT PARTITION
#SBATCH --partition=standard
# #SBATCH --partition=debug

## SET OUTPUT BASED ON SCRIPT NAME
#SBATCH --output=%x.job-%j.out
#SBATCH --error=%x.job-%j.out

# # enable email notification of job completion
# #SBATCH --mail-type=ALL
# #SBATCH --mail-user=profk@umd.edu

# options to pass to mpirun, no extras needed on Zaratan if using
# class configuration
MPIOPTS=""
# MPIOPTS+=" --mca mca_base_component_show_load_errors 0"

# Static parameters for all runs of kmeans
DATADIR="mnist-data"
NCLUST=20
MAXITERS=500
mkdir -p outdirs   # subdir for all output kmeans output

# # Full performance benchmark of all combinations of data files and
# # processor counts
# ALLDATA="digits_all_5e3.txt digits_all_1e4.txt digits_all_3e4.txt"
# ALLNP="1 2 4 8 10 16 32 64 128"

# Small sizes for testing
ALLDATA="digits_all_3e4.txt"
ALLNP="1 16 128"

date
hostname

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for DATA in $ALLDATA; do
        echo KMEANS $DATA with $NP procs
        OUTDIR=outdirs/outdir_${DATA}_${NP}
        /usr/bin/time -f "runtime: procs $NP data $DATA realtime %e" \
                      mpirun $MPIOPTS -np $NP ./kmeans_mpi $DATADIR/$DATA $NCLUST $OUTDIR $MAXITERS
        echo
    done
done

date
