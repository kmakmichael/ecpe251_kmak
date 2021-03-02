/*
Michael Kmak
ECPE 251 - High-Performance Computing
PA3 - Canny Edge Detector

usage: ./canny <image path> <sigma> <num threads>

*/

#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "math.h"
#include "sys/time.h"
#include "mpi.h"
#include "omp.h"

#include "sort.h"
#include "image_template.h"

#define timing_mode 1

// structs
typedef struct {
    float *data;
    int width;
    int height;
} img_s;

typedef struct {
    float *data;
    int w;
    int g; // length of ghost rows
    int d; // length of data
} chunk_s;

typedef struct {
    float *data;
    size_t w;
} kern_s;

// functions
void gaussian_kern(kern_s *kern, float sigma, float a);
void gaussian_deriv(kern_s *kern, float sigma, float a);
void kerninit(kern_s *kern);
void img_prep(const img_s *orig, img_s *cpy);
void chunk_prep(const chunk_s *base, chunk_s *cpy);
void h_conv(const chunk_s *in_img, chunk_s *out_img, const kern_s *kern);
void v_conv(const chunk_s *in_img, chunk_s *out_img, const kern_s *kern);
void suppression(const chunk_s *direction, const chunk_s *magnitude, chunk_s *out_img);
void hysteresis(chunk_s *hyst, float t_high, float t_low);
void edge_linking(const chunk_s *hyst, chunk_s *edges);
float timecalc(struct timeval start, struct timeval end);
void gather_and_save(const img_s *image, const chunk_s *chunk, char *filename);
void ghost_exchange(chunk_s *chunk);

int main(int argc, char *argv[]) {

    // rank 0 only
    img_s image;
    struct timeval start, compstart, conv, mag, sup, sort, doublethresh, compend, end;

    // all ranks
    chunk_s orig, temp, hori, vert, direction, magnitude, supp, hyst;
    kern_s h_kern, v_kern, h_deriv, v_deriv;
    float sigma;
    float a;
    size_t threadcount;
    int comm_size;
    int comm_rank;
    int rc;

    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        fprintf(stderr, "something bad i dont know\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    // argparse
    sigma = atof(argv[2]);
    threadcount = 1; //atoi(argv[3]);
    
    // check args
    if (!comm_rank) {
        if (sigma <= 0) {
            fprintf(stderr, "invalid sigma: %s\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
            return -1;
        }
        if (threadcount <= 0) {
            fprintf(stderr, "invalid number of OMP Threads: %s\n", argv[3]);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_ARG);
            return -1;
        } 
        /*
        if (numthreads > omp_get_num_procs()) {
            fprintf(stderr, "trying to use more threads than processors\n");
            MPI_Abort(MPI_COMM_WORLD, rc);
            return -1;
        }*/
    }

    //omp_set_num_threads(numthreads);
    
    if (!comm_rank) {
        read_image_template(argv[1], &image.data, &image.width, &image.height);
        // begin time
        gettimeofday(&start, NULL);
    }
    
    a = round(2.5 * sigma - 0.5);
    
    MPI_Bcast(&image.width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&image.height, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
    // calc chunk size
    orig.w = image.width;
    orig.d = image.height / comm_size;
    orig.g = ceil(a/2.0);
    
    orig.data = (float *) calloc((orig.d + 2*orig.g) * orig.w, sizeof(float));
 
    // prep all chunks
    chunk_prep(&orig, &temp);
    chunk_prep(&orig, &hori);
    chunk_prep(&orig, &vert);
    chunk_prep(&orig, &direction);
    chunk_prep(&orig, &magnitude);
    chunk_prep(&orig, &supp);
    chunk_prep(&orig, &hyst);
 
    rc = MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatter(image.data, image.width * orig.d, MPI_FLOAT, 
        &orig.data[orig.w * orig.g], orig.w * orig.d, MPI_FLOAT, 0, MPI_COMM_WORLD);
    ghost_exchange(&orig);
 
    h_kern.w = 2 * a + 1;
    h_kern.data = (float*) calloc(h_kern.w, sizeof(float));
    h_deriv.w = 2 * a + 1;
    h_deriv.data = (float*) calloc(h_deriv.w, sizeof(float));
    v_kern.w = 2 * a + 1;
    v_kern.data = (float*) calloc(v_kern.w, sizeof(float));
    v_deriv.w = 2 * a + 1;
    v_deriv.data = (float*) calloc(v_deriv.w, sizeof(float));

    gaussian_kern(&h_kern, sigma, a);
    gaussian_kern(&v_kern, sigma, a);
    gaussian_deriv(&h_deriv, sigma, a);
    gaussian_deriv(&v_deriv, sigma, a);

    if (!comm_rank)
        gettimeofday(&compstart, NULL);
    
    // horizontal
    h_conv(&orig, &temp, &h_kern);
    ghost_exchange(&temp);
    h_conv(&temp, &hori, &h_deriv);

    //vertical
    v_conv(&orig, &temp, &v_kern);
    ghost_exchange(&temp);
    v_conv(&temp, &vert, &v_deriv);
    
    if (!comm_rank)
        gettimeofday(&conv, NULL);

    // direction and magnitude
    ghost_exchange(&vert);
    ghost_exchange(&hori);
    for (int i = magnitude.g * magnitude.w; i < (magnitude.g + magnitude.d) * magnitude.w; i++) {
        magnitude.data[i] = sqrt((hori.data[i] * hori.data[i]) + (vert.data[i] * vert.data[i]));
    }
    for (int i = direction.g * direction.w; i < (direction.g + direction.d) * direction.w; i++) {
        direction.data[i] = atan2(hori.data[i], vert.data[i]);
    }

    if (!comm_rank)
        gettimeofday(&mag, NULL);

    ghost_exchange(&magnitude);
    ghost_exchange(&direction);
    suppression(&direction, &magnitude, &supp);

    if (!comm_rank)
        gettimeofday(&sup, NULL);
    
    MPI_Gather(&supp.data[supp.w * supp.g], supp.d * supp.w, MPI_FLOAT, 
        image.data, supp.d * supp.w, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float t_high;
    if (!comm_rank) {
        mergeSort(image.data, image.width * image.height, threadcount);
        t_high = image.data[(size_t) (image.height * image.width * 0.9)];
    }
 
    if (!comm_rank)
        gettimeofday(&sort, NULL);

    MPI_Bcast(&t_high, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
    ghost_exchange(&supp);
    memcpy(temp.data, supp.data, sizeof(float) * (supp.d + 2*supp.g) * supp.w);
    //ghost_exchange(&temp); 
    hysteresis(&temp, t_high, t_high/5.0);

    if (!comm_rank)
        gettimeofday(&doublethresh, NULL);

    ghost_exchange(&temp);
    edge_linking(&temp, &hyst);
    
    // stop time
    if (!comm_rank)
        gettimeofday(&compend, NULL);
   
    /*gather_and_save(&image, &orig, "original.pgm");
    gather_and_save(&image, &hori, "horizontal.pgm");
    gather_and_save(&image, &vert, "vertical.pgm");
    gather_and_save(&image, &direction, "direction.pgm");
    gather_and_save(&image, &magnitude, "magnitude.pgm");
    gather_and_save(&image, &supp, "suppression.pgm");*/
    gather_and_save(&image, &hyst, "output.pgm");

    if (!comm_rank) {
        gettimeofday(&end, NULL);

        printf("%d, %.1f, %zu, %.1f, %.1f\n", 
            image.height, 
            sigma, 
        threadcount, 
            timecalc(compstart, end), 
            timecalc(start, end)
        );

        if (timing_mode) {
            printf("\ncomp\tconv\tmag\tsup\tsort\tdt\tedge\ttotal\n");
            printf("%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f\n",
                timecalc(compstart, compend), // comp_time
                timecalc(compstart, conv), // conv_time
                timecalc(conv, mag), // mag_time
                timecalc(mag, sup), // sup_time
                timecalc(sup, sort), // sort_time
                timecalc(sort, doublethresh), // doublethresh_time
                timecalc(doublethresh, compend), // edge_time
                timecalc(start, end) // total_time
            );
        }
    }
    
    if (!comm_rank)
        free(image.data);
    
    free(orig.data);
    free(temp.data);
    free(hori.data);
    free(vert.data);
    free(magnitude.data);
    free(direction.data);
    free(supp.data);
    free(hyst.data);
    free(h_kern.data);
    free(v_kern.data);
    free(h_deriv.data);
    free(v_deriv.data);

    MPI_Finalize();

    return 0;
}

void img_prep(const img_s *orig, img_s *cpy) {
    cpy->height = orig->height;
    cpy->width = orig->width;
    cpy->data = (float *) calloc(cpy->height * cpy->width, sizeof(float));
}


void chunk_prep(const chunk_s *orig, chunk_s *cpy) {
    cpy->w = orig->w;
    cpy->d = orig->d;
    cpy->g = orig->g;
    cpy->data = (float *) calloc((orig->d + 2*orig->g) * orig->w, sizeof(float));
}



void print_kern(kern_s *kern) {
    for(size_t i = 0; i < kern->w; i++) {
        printf("[%f]", kern->data[i]);
    }
    printf("\n");
}

void gaussian_kern(kern_s *kern, float sigma, float a) {
    float sum = 0;

    kern->data = (float*) calloc(kern->w, sizeof(float));
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] = exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum += kern->data[i];
    }
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] /= sum;
    }
}

void gaussian_deriv(kern_s *kern, float sigma, float a) {
    float sum = 0;

    kern->data = (float*) calloc(kern->w, sizeof(float));

    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] = -1.0 * (i-a) * exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum -= i * kern->data[i];
    }
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] /= sum;
    }
    // kernel flipping
    for (size_t i = 0; i < (kern->w/2); i++) {
        float temp = kern->data[kern->w - 1 - i];
        kern->data[kern->w - 1 - i] = kern->data[i];
        kern->data[i] = temp;
    }
}

void h_conv(const chunk_s *in_img, chunk_s *out_img, const kern_s *kern) {
    int bounds = in_img->w * (in_img->d + in_img->g);
    int i = 0; // private when ||ized
    int offset = 0;
    for (int base = (in_img->g*in_img->w); base < bounds; base++) {
        float sum = 0;
        for (int k = 0; k < kern->w; k++) {
            offset = k - floor(kern->w / 2);
            i = base + offset;
            if (i / in_img->w == base / in_img->w) { // same row
                sum += in_img->data[i] * kern->data[k];
            }
        }
        out_img->data[base] = sum;
    }
}


void v_conv(const chunk_s *in_img, chunk_s *out_img, const kern_s *kern) {
    size_t bounds = in_img->w * (in_img->d + in_img->g);
    int i = 0; // private when ||ized
    int offset = 0;
    for (int base = (in_img->g*in_img->w); base < bounds; base++) {
        float sum = 0;
        for(int k = 0; k < kern->w; k++) {
            offset = (k - floor(kern->w / 2)) * in_img->w;
            i = base + offset;
            sum += in_img->data[i] * kern->data[k];
        }
        out_img->data[base] = sum;
    }
}

void suppression(const chunk_s *direction, const chunk_s *magnitude, chunk_s *supp) {
    #define Gxy magnitude->data
    size_t bounds = (direction->g + direction->d) * direction->w;
    size_t width = magnitude->w;
    size_t btm_right = width + 1;
    size_t btm_left = width - 1;
    float theta; // private when ||ized
    for (size_t i = direction->g * direction->w; i < bounds; i++) {
        theta = direction->data[i];
        if (theta < 0) {
            theta += M_PI;
        }
        theta *= (180.0 / M_PI);
        supp->data[i] = Gxy[i];
        // because of ghost rows, top and bottom checks can be removed
        if (theta <= 22.5 || theta > 157.5) {
            // top
            if (Gxy[i] < Gxy[i - width]) {
                supp->data[i] = 0;
            }
            // bottom
            if (Gxy[i] < Gxy[i + width]) {
                supp->data[i] = 0;
            }
        } else if (theta > 22.5 && theta <= 67.5) {
            //topleft
            if (i % width > 0) {
                if (Gxy[i] < Gxy[i - btm_right]) {
                    supp->data[i] = 0;
                }
            }
            // bottomright
            if (i % width < width-1) {
                if (Gxy[i] < Gxy[i + btm_right]) {
                    supp->data[i] = 0;
                }
            }
        } else if (theta > 67.5 && theta <= 112.5) {
            // left
            if (i % width > 0) {
                if (Gxy[i] < Gxy[i - 1]) {
                    supp->data[i] = 0;
                }
            }
            // right
            if (i % width < width-1) {
                if (Gxy[i] < Gxy[i + 1]) {
                    supp->data[i] = 0;
                }
            }
        } else if (theta > 112.5 && theta <= 157.5) {
            // topright
            if (i % width < width-1) {
                if (Gxy[i] < Gxy[i - btm_left]) {
                    supp->data[i] = 0;
                }
            }
            // bottomleft
            if (i % width > 0) {
                if (Gxy[i] < Gxy[i + btm_left]) {
                    supp->data[i] = 0;
                }
            }
        }
    }
    #undef Gxy
}


void hysteresis(chunk_s *hyst, float t_high, float t_low) {
    size_t bounds = hyst->w * (hyst->g + hyst->d);
    for (size_t i = hyst->g * hyst->w; i < bounds; i++) {
        if (hyst->data[i] >= t_high) {
            hyst->data[i] = 255;
        } else if (hyst->data[i] <= t_low) {
            hyst->data[i] = 0;
        } else {
            hyst->data[255] = 125;
        }
    }
}



void edge_linking(const chunk_s *hyst, chunk_s *edges) {
    size_t bounds = (hyst->g + hyst->d) * hyst->w;
    size_t width = hyst->w;
    size_t btm_right = width + 1;
    size_t btm_left = width - 1;
    for (size_t i = hyst->g * hyst->w; i < bounds; i++) {
        if(hyst->data[i] == 125) {
            edges->data[i] = 0;
            // bottom
            if (hyst->data[i + width] == 255) {
                edges->data[i] = 255;
            }
            // top
            if (hyst->data[i - width] == 255) {
                edges->data[i] = 255;
            }
            if (i % width > 0) {
                // left
                if (hyst->data[i - 1] == 255) {
                    edges->data[i] = 255;
                }
                // topleft
                if (hyst->data[i - btm_right] == 255) {
                    edges->data[i] = 255;
                }
                // bottomleft
                if (hyst->data[i + btm_left] == 255) {
                    edges->data[i] = 255;
                }
            }
            if (i % width > width-1) {
                // right
                if (hyst->data[i + 1] == 255) {
                    edges->data[i] = 255;
                }
                // topright
                if (hyst->data[i - btm_left] == 255) {
                    edges->data[i] = 255;
                }
                // bottomright
                if (hyst->data[i + btm_right] == 255) {
                    edges->data[i] = 255;
                }
            }
        } else {
            edges->data[i] = hyst->data[i];
        }
    }
}

float timecalc(struct timeval start, struct timeval end) {
    float ns = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    return ns / 1000.0;
}


void gather_and_save(const img_s *image, const chunk_s *chunk, char *filename) {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    MPI_Gather(&chunk->data[chunk->w * chunk->g], chunk->d * chunk->w, MPI_FLOAT, 
        image->data, chunk->d * chunk->w, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (!comm_rank)
        write_image_template(filename, image->data, image->width, image->height);
}


void ghost_exchange(chunk_s *chunk) {
    MPI_Status status;
    
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // send bottom downwards
    if (rank != size - 1) {
        MPI_Send(&chunk->data[chunk->d * chunk->w], chunk->w * chunk->g, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
    }
    if (rank) {
        MPI_Recv(chunk->data, chunk->w * chunk->g, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &status);
    }

    // send top upwards
    if (rank) {
        MPI_Send(&chunk->data[chunk->g * chunk->w], chunk->w * chunk->g, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
    }    
    if (rank != size - 1) {
        MPI_Recv(&chunk->data[(chunk->g + chunk->d) * chunk->w], chunk->w * chunk->g, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, &status);
    }
}
