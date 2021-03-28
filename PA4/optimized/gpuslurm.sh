#!/bin/bash
##SBATCH --partition=compute    ### Partition
#SBATCH --job-name=PA4          ### Job Name
#SBATCH --time=03:00:00         ### WallTime
#SBATCH --nodes=1               ### Number of Nodes
#SBATCH --tasks-per-node=1      ### Number of tasks

rm Serial.csv # clear previous results

for sig in 0.6 1.1
do
    for ((i=1024;i<=8192;i=i*2)) do
        for((j=0;j<5;j++)) do
	        srun ./canny ~/lennas/Lenna_org_$i.pgm $sig >> Serial.csv
        done
    done
    for((j=0;j<5;j++)) do
	    srun ./canny ~/lennas/Lenna_org_10240.pgm $sig >> Serial.csv
    done
done
