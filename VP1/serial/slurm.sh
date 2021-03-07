#!/bin/bash
#SBATCH --partition=compute
#SBATCH --job-name=VP1-S
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --nodelist=node006
rm Serial.csv
for img in 256 512 1024 2048 3072
do
    for((i=0;i<5;i++)) do
        srun ./adf lennas/Lenna_org_$img.pgm lennas/Lenna_org_$img.pgm >>Serial.csv
    done
done
