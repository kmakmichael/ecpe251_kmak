/*
    Michael Kmak
    ECPE 251 - High-Performance Computing
    PA6 - Performance Prediction

    usage: ./memcpy size

    copies a randomly-filled float array between
    the CPU and GPU in both directions. size of
    this array is 2^N bytes, where N is given as
    a program argument. outputs time taken in 
    the following format:
        size H2D D2H
*/

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>

#define GPU_NO 1 // 85 % 4


int main(int argc, char *argv[]) {
    
    float *h_arr;
    float *d_arr;
    size_t n;

    // argparse
    if (argc != 2) {
        fprintf(stderr, "usage: ./memcpy <size>\n");
        return 1;
    }
    int success = sscanf(argv[1], "%zu", &n);
    if (success != 1) {
        fprintf(stderr, "invalid size, enter an integer\n");
        return -1;
    }
    if (n > log2((float)SIZE_MAX)) {
        fprintf(stderr, "enter a power of two no larger than %f\n", log2((float)SIZE_MAX));
        return -1;
    }

    printf("Creating array of floats. size 2^%zu=%f bytes, containing %f floats\n", n, exp2((float)n), exp2((float)n)/sizeof(float));
    n = exp2((float)n);
}
