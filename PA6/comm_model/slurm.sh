#!/bin/bash
##SBATCH --partition=compute    ### Partition
#SBATCH --job-name=memcpy       ### Job Name
#SBATCH --time=03:00:00         ### WallTime
#SBATCH --nodes=1               ### Number of Nodes
#SBATCH --tasks-per-node=1      ### Number of tasks

rm memcpy.csv # clear previous results

echo "size, htod, dtoh" >> memcpy.csv
for((size=10;size<30;size++)) do
    for((j=0;j<30;j++)) do
	    srun ./memcpy $size >> memcpy.csv
    done
done
