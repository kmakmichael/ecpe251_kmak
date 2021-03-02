#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=PA3        ### Job Name
#SBATCH --time=03:00:00       ### WallTime
#SBATCH --nodes=4             ### Number of Nodes
#SBATCH --tasks-per-node=8   ### Number of tasks (MPI processes)
srun ./canny images/Lenna_org_2048.pgm 0.6 4  > testruns.csv
