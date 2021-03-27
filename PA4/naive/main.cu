/*
    Michael Kmak    
    ECPE 251 - High-Performance Computing
    PA4 - Canny Edge Sate 1 - GPU

    usage: ./canny <image path> <sigma>

*/

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

//#include "sort.h"
#include "image_template.h"

#define GPU_NO 1 // 85 % 4

int main(int argc, char *argv[]) {

    float sigma;
    if (argc != 3) {
        fprintf(stderr, "usage: ./canny <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
        return -1;
    }

    cudaSetDevice(GPU_NO);
}
