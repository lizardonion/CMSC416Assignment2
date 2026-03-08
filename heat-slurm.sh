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

# No output to screen, timing only, 500 simulation steps
OUTPUT=0
STEPS=500

# ## Full performance run, problem data for rod width and number of procs
# ALLWIDTHS="6400 25600 102400 204800"
# ALLNP="1 2 4 8 10 16 32 64 128" 

# ## Smaller testing runs
# ALLWIDTHS="6400 25600"
# ALLNP="1 16 64 128" 

date
hostname

# Iterate over all proc/data file combos
for NP in $ALLNP; do 
    for WIDTH in $ALLWIDTHS; do
        echo HEAT $STEPS $WIDTH with $NP procs
        /usr/bin/time -f "runtime: procs $NP width $WIDTH realtime %e" \
                      mpirun $MPIOPTS -np $NP ./heat_mpi $STEPS $WIDTH $OUTPUT
        echo
    done
done

date
