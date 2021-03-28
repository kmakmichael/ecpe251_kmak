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
#define BLOCKSIZE 16

void print_k(float *k, int len);
void g_kern(float *k, float sigma);
void g_deriv(float *k, float sigma);

int main(int argc, char *argv[]) {

    int height;
    int width;
    float sigma;
    int kern_w;

    // host
    float *h_img;
    //float *h_mag;
    //float *h_dir;
    float *h_vkern;
    float *h_hkern;
    float *h_vderiv;
    float *h_hderiv;

    // device
    /*float *d_img;
    float *d_temp;
    float *d_hori;
    float *d_vert;
    float *d_mag;
    float *d_dir;
    float *d_vkern;
    float *d_hkern;
    float *d_vderiv;
    float *d_hderiv;
    */
 
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

    read_image_template(argv[1], &h_img, &width, &height);

    // prepare kernels
    kern_w = 2 * round(2.5 * sigma - 0.5) + 1;
    
    h_vkern = (float *) calloc(kern_w, sizeof(float));
    h_hkern = (float *) calloc(kern_w, sizeof(float));
    h_vderiv = (float *) calloc(kern_w, sizeof(float));
    h_hderiv = (float *) calloc(kern_w, sizeof(float));

    g_kern(h_vkern, sigma);
    g_kern(h_hkern, sigma);
    g_deriv(h_vderiv, sigma);
    g_deriv(h_hderiv, sigma);
    print_k(h_vkern, kern_w);
    print_k(h_hkern, kern_w);
    print_k(h_vderiv, kern_w);
    print_k(h_hderiv, kern_w);

    // transfer kernels


    // free
    free(h_vkern);
    free(h_hkern);
    free(h_vderiv);
    free(h_hderiv);
}


void print_k(float *k, int len) {
    for (size_t i = 0; i < len; i++) {
        printf("[%f]", k[i]);
    }
    printf("\n");
}


void g_kern(float *k, float sigma) {
    float a = round(2.5 * sigma - 0.5);
    int w = 2 * a + 1;
    float sum = 0;

    for (size_t i = 0; i < w; i++) {
        k[i] = exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum += k[i];
    }
    for (size_t i = 0; i < w; i++) {
        k[i] /= sum;
    }
}


void g_deriv(float *k, float sigma) {
    float a = round(2.5 * sigma - 0.5);
    int w = 2 * a + 1;
    float sum = 0;
    
    for (size_t i = 0; i < w; i++) {
        k[i] = -1.0 * (i-a) * exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum -= i * k[i];
    }
    for (size_t i = 0; i < w; i++) {
        k[i] /= sum;
    }
    // flip
    for (size_t i = 0; i < (w/2); i++) {
        float temp = k[w-1-i];
        k[w-1-i] = k[i];
        k[i] = temp;
    }
}















