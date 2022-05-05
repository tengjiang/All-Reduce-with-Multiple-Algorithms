#!/bin/bash
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH -t 5:00

mpiexec -n 8 ../slurm_outputs/allreduce_ring
