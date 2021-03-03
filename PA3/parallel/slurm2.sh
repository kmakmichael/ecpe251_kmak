#!/bin/bash
#SBATCH --partition=compute   ### Partition
#SBATCH --job-name=PA3-2	  ### Job Name
#SBATCH --time=03:00:00       ### WallTime
#SBATCH --nodes=1             ### Number of Nodes
#SBATCH --ntasks-per-node=2   ### Number of tasks (MPI processes)

rm Parallel_2.csv
for sig in 0.6 1.1
do
	for((i=1024;i<=8192;i=i*2)) do #each image 1024, 2048, 4096, 8192
		for((j=0;j<30;j++)) do	#each image executed 30 times
			srun ./canny images/Lenna_org_$i.pgm $sig 4 >>Parallel_2.csv
		done
	done

	for((j=0;j<30;j++)) do  #each image executed 30 times
		srun ./canny images/Lenna_org_10240.pgm $sig 4 >>Parallel_2.csv
	done

	for((j=0;j<30;j++)) do  #each image executed 30 times
		srun ./canny images/Lenna_org_12800.pgm $sig 4 >>Parallel_2.csv
	done
done
