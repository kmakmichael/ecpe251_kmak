#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=VP1-P
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node005
rm Parallel.csv

for t in 4 8 16
do
for img in 256 512 1024 2048 3072
do
    for((i=0;i<5;i++)) do
        srun ./adf lennas/Lenna_org_$img.pgm lennas/Lenna_org_$img.pgm $t >>Parallel.csv
    done
done
done
