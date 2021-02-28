#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=PA3		  ### Job Name
#SBATCH --time=03:00:00       ### WallTime
#SBATCH --nodes=4             ### Number of Nodes
#SBATCH --ntasks-per-node=4   ### Number of tasks (MPI processes)
#SBATCH --nodelist=node005,node006,node007,node008

for sig in 0.6 1.1 
do
	for((i=1024;i<=8192;i=i*2)) do #each image 1024, 2048, and 4096
		for((j=0;j<30;j++)) do	#each image executed 30 times
			srun /canny images/Lenna_org_$i.pgm $sig $t>>Parallel.csv
		done
	done

	for((j=0;j<30;j++)) do  #each image executed 30 times
		srun ./canny images/Lenna_org_7680.pgm $sig $t>>Parallel.csv
	done
    
	for((j=0;j<30;j++)) do  #each image executed 30 times
		srun ./canny images/Lenna_org_10240.pgm $sig $t>>Parallel.csv
	done

	for((j=0;j<30;j++)) do  #each image executed 30 times
		srun ./canny images/Lenna_org_12800.pgm $sig $t>>Parallel.csv
	done
done
