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
#include <math_constants.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

#include "image_template.h"

#define GPU_NO 1 // 85 % 4
#define blocksize 8
#define conv_size 512

#define debug_mode


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


__global__
void gpu_suppression(float *Gxy, float *dir, float *supp, int height, int width) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y; 
    int k = i*width + j;
    size_t bounds = width * height;
    size_t btm_right = width + 1;
    size_t btm_left = width - 1;
    float suppval;    

    float theta = dir[k];
    if (theta <= 0)
        theta += CUDART_PI_F;
    theta *= (180.0 / CUDART_PI_F);
    suppval = Gxy[k];
    if (theta <= 22.5 || theta > 157.5) {
        // top
        if (k >= width) {
            if (Gxy[k] < Gxy[k - width]) {
                suppval = 0;
            }
        }
        // bottom
        if (k < bounds - width) {
            if (Gxy[k] < Gxy[k + width]) {
                suppval = 0;
            }
        }
    } else if (theta > 22.5 && theta <= 67.5) {
        //topleft
        if (k >= width && k % width > 0) {
            if (Gxy[k] < Gxy[k - btm_right]) {
                suppval = 0;
            }
        }
        // bottomright
        if (k < bounds - width && k % width < width-1) {
            if (Gxy[k] < Gxy[k + btm_right]) {
                suppval = 0;
            }
        }
    } else if (theta > 67.5 && theta <= 112.5) {
        // left
        if (k % width > 0) {
            if (Gxy[k] < Gxy[k - 1]) {
                suppval = 0;
            }
        }       
        // right
        if (k % width < width-1) {
            if (Gxy[k] < Gxy[k + 1]) {
                suppval = 0;
            }
        }
    } else if (theta > 112.5 && theta <= 157.5) {
        // topright
        if (k >= width && k % width < width-1) {
            if (Gxy[k] < Gxy[k - btm_left]) {
                suppval = 0;
            }
        }
        // bottomleft
        if (k < bounds - width && k % width > 0) {
            if (Gxy[k] < Gxy[k + btm_left]) {
                suppval = 0;
            }
        }
    }
    supp[k] = suppval;
}


__global__
void gpu_hysteresis(float *hyst, float width, float t_hi, float t_lo) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int k = i*width+j;

    if (hyst[k] >= t_hi) {
        hyst[k] = 255.0;
    } else if (hyst[k] <= t_lo) {
        hyst[k] = 0.0;
    } else {
        hyst[k] = 125.0;
    }
}


__global__
void gpu_edgelinking(float *hyst, float *edge, int width, int height) { 
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    int j = threadIdx.y + blockIdx.y*blockDim.y;
    int k = i*width+j;
    int bounds = width * height;
    int btm_right = width + 1;
    int btm_left = width - 1;
    int edgeval = 0;

    if(hyst[k] == 125) {
        edge[k] = 0;
        // topleft
        if (k >= width && k % width > 0) {
            if (hyst[k - btm_right] == 255) {
                edge[k] = 255;
            }
        }
        // top
        if (k >= width) {
            if (hyst[k - width] == 255) {
                edge[k] = 255;
            }
        }
        // topright
        if (k >= width && k % width < width-1) {
            if (hyst[k - btm_left] == 255) {
                edge[k] = 255;
            }
        }
        // left
        if (k % width > 0) {
            if (hyst[k - 1] == 255) {
                edge[k] = 255;
            }
        }
        // right
        if (k % width > width-1) {
            if (hyst[k + 1] == 255) {
                edge[k] = 255;
            }
        }
        // bottomleft
        if (k < bounds - width && k % width > 0) {
            if (hyst[k + btm_left] == 255) {
                edge[k] = 255;
            }
        }
        // bottom
        if (k < bounds - width) {
            if (hyst[k + width] == 255) {
                edge[k] = 255;
            }
        }
        // bottomright
        if (k < bounds - width && k % width < width-1) {
            if (hyst[k + btm_right] == 255) {
                edge[k] = 255;
            }
        }
    } else {
        edge[k] = hyst[k];
    }
}


int main(int argc, char *argv[]) {

    int height;
    int width;
    float sigma;
    int kern_w;
    struct timeval start, stop, compstart, compend;
    #ifdef debug_mode
    float commtime, convtime, magdirtime, supptime, sorttime, hysttime, edgetime;
    #endif

    // host
    float *h_img;
    float *h_mag;
    float *h_dir;
    float *h_supp;
    float *h_hyst;
    float *h_edge;
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
    float *d_supp;
    float *d_hyst;
    float *d_edge;
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
    h_supp = (float *) calloc(width*height, sizeof(float));
    h_hyst = (float *) calloc(width*height, sizeof(float));
    h_edge = (float *) calloc(width*height, sizeof(float));
    cudaMalloc((void **)&d_img, sizeof(float)*width*height);
    cudaMalloc((void **)&d_temp, sizeof(float)*width*height);
    cudaMalloc((void **)&d_hori, sizeof(float)*width*height);
    cudaMalloc((void **)&d_vert, sizeof(float)*width*height);
    cudaMalloc((void **)&d_mag, sizeof(float)*width*height);
    cudaMalloc((void **)&d_dir, sizeof(float)*width*height);
    cudaMalloc((void **)&d_supp, sizeof(float)*width*height);
    cudaMalloc((void **)&d_hyst, sizeof(float)*width*height);
    cudaMalloc((void **)&d_edge, sizeof(float)*width*height);

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
    #ifdef debug_mode
    gettimeofday(&start, NULL);
    #endif
    cudaMemcpy(d_vkern, h_vkern, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hkern, h_hkern, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vderiv, h_vderiv, sizeof(float)*kern_w, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hderiv, h_hderiv, sizeof(float)*kern_w, cudaMemcpyHostToDevice);

    // transfer image
    cudaMemcpy(d_img, h_img, sizeof(float)*width*height, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    commtime = timecalc(start, stop);
    #endif

    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&start, NULL);
    #endif

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
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    convtime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif

    //mag & dir
    dim3 dB(blocksize, blocksize);
    dim3 dG(width/blocksize, height/blocksize);
    gpu_magdir<<<dG,dB>>>(d_hori, d_vert, d_mag, d_dir, height, width);
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    magdirtime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif

    // suppression
    gpu_suppression<<<dG,dB>>>(d_mag, d_dir, d_supp, height, width);
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    supptime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif

    // sorting
    cudaMemcpy(d_hyst, d_supp, sizeof(float)*height*width, cudaMemcpyDeviceToDevice);
    thrust::device_ptr<float> thr_d(d_hyst);
    thrust::device_vector<float> d_hyst_vec(thr_d,thr_d+(height*width));
    thrust::sort(d_hyst_vec.begin(),d_hyst_vec.end());
    int index = (int) (0.9 * height*width);
    float t_hi = d_hyst_vec[index];
    float t_lo = t_hi * 0.2;
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    sorttime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif


    // hysteresis
    gpu_hysteresis<<<dG,dB>>>(d_hyst, width, t_hi, t_lo);
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    hysttime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif


    // edge linking
    gpu_edgelinking<<<dG,dB>>>(d_hyst, d_edge, height, width);
    cudaDeviceSynchronize();
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    edgetime = timecalc(start, stop);
    gettimeofday(&start, NULL);
    #endif

    // pull results
    #ifdef debug_mode 
    cudaMemcpy(h_mag, d_mag, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dir, d_dir, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_supp, d_supp, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hyst, d_hyst, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    #endif
    cudaMemcpy(h_img, d_edge, sizeof(float)*width*height, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); 
    #ifdef debug_mode
    gettimeofday(&stop, NULL);
    commtime += timecalc(start, stop);
    #endif

    // computation end
    gettimeofday(&compend, NULL);

    // write results
    #ifdef debug_mode
    write_image_template<float>("magnitude.pgm", h_mag, width, height);
    write_image_template<float>("direction.pgm", h_dir, width, height);
    write_image_template<float>("suppression.pgm", h_supp, width, height);
    write_image_template<float>("hysteresis.pgm", h_hyst, width, height);
    #endif
    write_image_template<float>("out.pgm", h_img, width, height);

    #ifndef debug_mode
    printf("%d, %0.3f\n", height, timecalc(compstart, compend)); 
    #else
    printf("idx=%d, hi=%0.2f, lo=%0.2f\n", index, t_hi, t_lo);
    printf("%0.3f,%0.3f,%0.3f,%0.3f,%0.3f, %0.3f, %0.3f, %0.3f\n",
        convtime,
        magdirtime,
        supptime,
        sorttime,
        hysttime,
        edgetime,
        commtime,
        timecalc(compstart, compend)
    );
    #endif

    // free
    free(h_vkern);
    free(h_hkern);
    free(h_vderiv);
    free(h_hderiv);
    free(h_mag);
    free(h_dir);
    free(h_supp);
    free(h_hyst);
    free(h_edge);
    cudaFree(d_img);
    cudaFree(d_temp);
    cudaFree(d_hori);
    cudaFree(d_vert);
    cudaFree(d_mag);
    cudaFree(d_dir);
    cudaFree(d_supp);
    cudaFree(d_hyst);
    cudaFree(d_edge);
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
