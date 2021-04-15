#!/bin/bash
##SBATCH --partition=compute    ### Partition
#SBATCH --job-name=memcpy       ### Job Name
#SBATCH --time=03:00:00         ### WallTime
#SBATCH --nodes=1               ### Number of Nodes
#SBATCH --tasks-per-node=1      ### Number of tasks

rm tout.csv # clear previous results

for((size=8;size<30;size++)) do
    srun ./memcpy $size >> tout.csv
done
