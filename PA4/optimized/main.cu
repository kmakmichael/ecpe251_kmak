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
#define blocksize 8
#define conv_size 512


void print_k(float *k, int len);
float timecalc(struct timeval start, struct timeval end);
void g_kern(float *k, float sigma);
void g_deriv(float *k, float sigma);


__global__
void gpu_hconvolve(float *img, float *out, int width, int height, float *kern, int kern_w) {
    extern __shared__ float smem[]; 
    float *s_kern = smem;
    float *s_img = &smem[kern_w];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y; 
    int localidx = threadIdx.x;
    int localidy = threadIdx.y;
    int globalidx = localidx + blockIdx.x*blockDim.x;
    int globalidy = localidy + blockIdx.y*blockDim.y;
    int base = i*width + j;
    int p;

    // load to shared mem
    if (localidy < kern_w)
        s_kern[localidy] = kern[localidy];
    s_img[localidy] = img[globalidx*width + globalidy];
    __syncthreads();

    if (i < height && j < width) {
        float sum = 0;
        for (size_t k = 0; k < kern_w; k++) {
            int offset = k - floorf(kern_w/2);
            p = base + offset;
            if (p / width == base / width) { // same row
                if (p >= 0 && p <= height * width) {
                    if (localidy + offset < blockDim.y)
                        sum += s_img[localidy + offset] * s_kern[k];
                    else
                        sum += img[p] * kern[k];
                }
            }
        }
        out[base] = sum;
    }
}


__global__
void gpu_vconvolve(float *img, float *out, int width, int height, float *kern, int kern_w) {
    extern __shared__ float smem[];
    float *s_kern = smem;
    float *s_img = &smem[kern_w];
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y; 
    int localidx = threadIdx.x;
    int localidy = threadIdx.y;
    int globalidx = localidx + blockIdx.x*blockDim.x;
    int globalidy = localidy + blockIdx.y*blockDim.y;
    int base = i*width + j;
    int p;

    // load to shared mem
    if (localidx < kern_w)
        s_kern[localidx] = kern[localidx];
    s_img[localidx] = img[globalidx*width + globalidy];
    __syncthreads();


    if (i < height && j < width) {
        float sum = 0;
        for (size_t k = 0; k < kern_w; k++) {
            int offset = (k - floorf(kern_w/2)) * width;
            p = base + offset;
            if (p >= 0 && p < width * height) {
                if (localidx + offset < blockDim.x) 
                    sum += s_img[localidx + offset] * s_kern[k];
                else
                    sum += img[p] * s_kern[k];
            }
        }
        out[base] = sum;
    }
}

__global__
void gpu_magdir(float *hori, float *vert, float *mag, float *dir, int height, int width) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int k = i*width+j;
    float ho = hori[k];
    float ve = vert[k];

    mag[k] = sqrtf((ho * ho) + (ve * ve));
    dir[k] = atan2f(ho, ve);
} 


int main(int argc, char *argv[]) {

    int height;
    int width;
    float sigma;
    int kern_w;
    struct timeval convstart, convstop, magdirstart, magdirstop, htodstart, htodstop, dtohstart, dtohstop, compstart, compstop;

    // host
    float *h_img;
    float *h_mag;
    float *h_dir;
    float *h_vkern;
    float *h_hkern;
    float *h_vderiv;
    float *h_hderiv;

    // device
    float *d_img;
    float *d_temp;
    float *d_hori;
    float *d_vert;
    float *d_mag;
    float *d_dir;
    float *d_vkern;
    float *d_hkern;
    float *d_vderiv;
    float *d_hderiv;
 
    // argparse
    if (argc != 3) {
        fprintf(stderr, "usage: ./canny <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
        return -1;
    }

    // cuda setup
    cudaSetDevice(GPU_NO);

    // image prep
    read_image_template<float>(argv[1], &h_img, &width, &height);
    h_mag = (float *) calloc(width*height, sizeof(float));
    h_dir = (float *) calloc(width*height, sizeof(float));
    cudaMalloc((void **)&d_img, sizeof(float)*width*height);
    cudaMalloc((void **)&d_temp, sizeof(float)*width*height);
    cudaMalloc((void **)&d_hori, sizeof(float)*width*height);
    cudaMalloc((void **)&d_vert, sizeof(float)*width*height);
    cudaMalloc((void **)&d_mag, sizeof(float)*width*height);
    cudaMalloc((void **)&d_dir, sizeof(float)*width*height);

    // computation start
    gettimeofday(&compstart, NULL);

    // prepare canny kernels
    kern_w = 2 * round(2.5 * sigma - 0.5) + 1;
    h_vkern = (float *) calloc(kern_w, sizeof(float));
    h_hkern = (float *) calloc(kern_w, sizeof(float));
    h_vderiv = (float *) calloc(kern_w, sizeof(float));
    h_hderiv = (float *) calloc(kern_w, sizeof(float));
    cudaMalloc((void **)&d_vkern, sizeof(float)*kern_w);
    cudaMalloc((void **)&d_hkern, sizeof(float)*kern_w);
    cudaMalloc((void **)&d_vderiv, sizeof(float)*kern_w);
    cudaMalloc((void **)&d_hderiv, sizeof(float)*kern_w);

    // calculate ckernels
    g_kern(h_vkern, sigma);
    g_kern(h_hkern, sigma);
    g_deriv(h_vderiv, sigma);
    g_deriv(h_hderiv, sigma);

    // transfer ckernels
    gettimeofday(&htodstart, NULL);
    cudaMemcpy(d_vkern, h_vkern, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hkern, h_hkern, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vderiv, h_vderiv, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hderiv, h_hderiv, sizeof(float)*kern_w, cudaMemcpyHostToDevice);

    // transfer image
    cudaMemcpy(d_img, h_img, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    gettimeofday(&htodstop, NULL);


    cudaDeviceSynchronize();
    gettimeofday(&convstart, NULL);

    // horizontal convolve
    dim3 h_dB(1, conv_size);
    dim3 h_dG(height, width/conv_size);
    int memsize = sizeof(float) * (conv_size + kern_w);
    gpu_hconvolve<<<h_dG,h_dB,memsize>>>(d_img, d_temp, width, height, d_hkern, kern_w);
    gpu_hconvolve<<<h_dG,h_dB,memsize>>>(d_temp, d_hori, width, height, d_hderiv, kern_w);

    // vertical convolve
    dim3 v_dB(conv_size, 1);
    dim3 v_dG(height/conv_size, width);
    gpu_vconvolve<<<v_dG,v_dB,memsize>>>(d_img, d_temp, width, height, d_vkern, kern_w);
    gpu_vconvolve<<<v_dG,v_dB,memsize>>>(d_temp, d_vert, width, height, d_vderiv, kern_w);
    cudaDeviceSynchronize();
    gettimeofday(&convstop, NULL);

    // mag & dir
    gettimeofday(&magdirstart, NULL);
    dim3 md_dB(blocksize, blocksize);
    dim3 md_dG(width/blocksize, height/blocksize);
    gpu_magdir<<<md_dG,md_dB>>>(d_hori, d_vert, d_mag, d_dir, height, width);
    cudaDeviceSynchronize();
    gettimeofday(&magdirstop, NULL);

    // pull results
    gettimeofday(&dtohstart, NULL);
    cudaMemcpy(h_mag, d_mag, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dir, d_dir, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    gettimeofday(&dtohstop, NULL);

    // computation end
    gettimeofday(&compstop, NULL);

    // write results
    write_image_template<float>("magnitude.pgm", h_mag, width, height);
    write_image_template<float>("direction.pgm", h_dir, width, height);
    printf("%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",
        timecalc(convstart, convstop),
        timecalc(magdirstart, magdirstop),
        timecalc(htodstart, htodstop),
        timecalc(dtohstart, dtohstop),
        timecalc(compstart, compstop)
    );

    // free
    free(h_vkern);
    free(h_hkern);
    free(h_vderiv);
    free(h_hderiv);
    free(h_mag);
    free(h_dir);
    cudaFree(d_img);
    cudaFree(d_temp);
    cudaFree(d_hori);
    cudaFree(d_vert);
    cudaFree(d_mag);
    cudaFree(d_dir);
}


void print_k(float *k, int len) {
    for (size_t i = 0; i < len; i++) {
        printf("[%f]", k[i]);
    }
    printf("\n");
}


float timecalc(struct timeval start, struct timeval end) {
    float ns = (end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec);
    return ns / 1000.0;
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
