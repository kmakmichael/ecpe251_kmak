/*
    Michael Kmak
    ECPE 251 - High-Performance Computing
    PA6 - Performance Prediction

    usage: ./memcpy size

    copies a randomly-filled float array between
    the CPU and GPU in both directions. size of
    this array is 2^n floats, where n is given as
    a program argument. outputs time taken in 
    the following format:
        size H2D D2H
*/

#include <cuda.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>

#define GPU_NO 1 // 85 % 4

float timecalc(struct timeval start, struct timeval end) {
    float us = (end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec);
    return us / 1000.0;
}

int main(int argc, char *argv[]) {
    
    float *h_arr;
    float *d_arr;
    size_t size;
    size_t n;

    struct timeval start, end;
    float htod, dtoh;

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

    size = exp2((float)n) * sizeof(float);
    h_arr = (float *) malloc(size);
    cudaMalloc((void **)&d_arr, size);

    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice); 
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    htod = timecalc(start, end);

    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    dtoh = timecalc(start, end);

    char sizeout[8];
    switch (n/10) {
        case 0:
            sprintf(sizeout, "%0.0f", exp2((float)(n%10)));
            break;
        case 1:
            sprintf(sizeout, "%0.0fK", exp2((float)(n%10)));
            break;
        case 2:
            sprintf(sizeout, "%0.0fM", exp2((float)(n%10)));
            break;
        case 3:
            sprintf(sizeout, "%0.0fG", exp2((float)(n%10)));
            break;
        default:
            sprintf(sizeout, "err");
    }
    printf("%s, %0.2f, %0.2f\n", sizeout, htod, dtoh);

    free(h_arr);
    cudaFree(d_arr);
}
