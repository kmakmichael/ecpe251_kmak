#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=ML_SNN	  ### Job Name
#SBATCH --time=03:00:00       ### WallTime
#SBATCH --nodes=1             ### Number of Nodes
#SBATCH --ntasks-per-node=4   ### Number of tasks (MPI processes)

for i in 240 480 960 1200 1440 1680 2160 2400
do
    for ((j=0;j<30;j++))
    do
        srun ./snn_morris $i 1 >> parallel_4.csv
    done
done
