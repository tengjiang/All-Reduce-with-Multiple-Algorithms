#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 2
#SBATCH --cpus-per-task=8
#SBATCH -t 5:00

mpiexec -n 5 ../slurm_outputs/allreduce_tree
