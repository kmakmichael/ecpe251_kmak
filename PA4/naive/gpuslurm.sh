#!/bin/bash
##SBATCH --partition=compute    ### Partition
#SBATCH --job-name=PA4          ### Job Name
#SBATCH --time=03:00:00         ### WallTime
#SBATCH --nodes=1               ### Number of Nodes
#SBATCH --tasks-per-node=1      ### Number of tasks

for((i=0;i<1;i++)) do
	srun ./canny ~/lennas/Lenna_org_4096.pgm
done
