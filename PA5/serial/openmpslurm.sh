#!/bin/bash
##SBATCH --partition=compute  ### Partition
#SBATCH --job-name=PA5_S      ### Job Name
#SBATCH --time=03:00:00       ### WallTime
##SBATCH --nodelist=node005    ### Node
#SBATCH --nodes=1             ### Number of Nodes
#SBATCH --ntasks-per-node=1   ### Number of tasks (MPI processes)

for i in 256, 512, 1024, 2048, 4096, 8192, 10240 do
    srun ./canny ~/lennas/Lenna_org_$i.pgm 0.6>>Serial.csv
done
