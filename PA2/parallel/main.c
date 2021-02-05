/*
    Michael Kmak
    ECPE 251 - High-Performance Computing
    PA1 - Canny Edge Detector

    usage: ./canny <image path> <sigma> <num threads>

*/

#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "math.h"
#include "sys/time.h"
#include "omp.h"

#include "sort.h"
#include "image_template.h"

#define timing_mode 0

// structs
typedef struct {
    float *data;
    int width;
    int height;
} img_s;

typedef struct {
    float *data;
    size_t w;
} kern_s;

// functions
void gaussian_kern(kern_s *kern, float sigma, float a);
void gaussian_deriv(kern_s *kern, float sigma, float a);
void kerninit(kern_s *kern);
void img_prep(const img_s *orig, img_s *cpy);
void h_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void v_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void suppression(img_s *direction, img_s *magnitude, img_s *out_img);
void hysteresis(img_s *img, float t_high, float t_low);
void edge_linking(img_s *hyst, img_s *edges);
float timecalc(struct timeval start, struct timeval end);

int sortcomp(const void *e1, const void *e2);

int main(int argc, char *argv[]) {

    img_s image;
    img_s temp;
    img_s vert;
    img_s hori;
    img_s magnitude;
    img_s direction;
    img_s supp;
    img_s hyst;
    size_t numthreads;
    kern_s h_kern;
    kern_s h_deriv;
    kern_s v_kern;
    kern_s v_deriv;
    struct timeval start, compstart, conv, mag, sup, sort, doublethresh, edge, compend, end;
    float sigma;
    float a;

    if (argc != 4) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma> <# threads>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
        return -1;
    }
    numthreads = atoi(argv[3]);
    if (numthreads <= 0) {
        fprintf(stderr, "invalid number of threads: %s\n", argv[3]);
        return -1;
    }
    if (numthreads > omp_get_num_procs()) {
        fprintf(stderr, "trying to use more threads than processors\n");
        return -1;
    }
    omp_set_num_threads(numthreads);

    // begin time
    gettimeofday(&start, NULL);
    read_image_template(argv[1], &image.data, &image.width, &image.height);
    #pragma omp parallel sections
    {
        #pragma omp section
        img_prep(&image, &temp);
        #pragma omp section
        img_prep(&image, &vert);
        #pragma omp section
        img_prep(&image, &hori);
        #pragma omp section
        img_prep(&image, &magnitude);
        #pragma omp section
        img_prep(&image, &direction);
        #pragma omp section
        img_prep(&image, &supp);
        #pragma omp section
        img_prep(&image, &hyst);
    }
    
    // kernel initialization
    a = round(2.5 * sigma - 0.5);
    #pragma omp parallel sections
    {
        #pragma omp section
        h_kern.w = 2 * a + 1;
        gaussian_kern(&h_kern, sigma, a);
        h_kern.data = (float*) calloc(h_kern.w, sizeof(float));
        #pragma omp section
        v_kern.w = 2 * a + 1;
        v_kern.data = (float*) calloc(v_kern.w, sizeof(float));
        gaussian_kern(&v_kern, sigma, a);
        #pragma omp section
        h_deriv.w = 2 * a + 1;
        h_deriv.data = (float*) calloc(h_deriv.w, sizeof(float));
        gaussian_deriv(&h_deriv, sigma, a);
        #pragma omp section
        v_deriv.w = 2 * a + 1;
        v_deriv.data = (float*) calloc(v_deriv.w, sizeof(float));
        gaussian_deriv(&v_deriv, sigma, a);
    }

    gettimeofday(&compstart, NULL);

    // horizontal
    h_conv(&image, &temp, &h_kern);
    h_conv(&temp, &hori, &h_deriv);

    // vertical
    v_conv(&image, &temp, &v_kern);
    v_conv(&temp, &vert, &v_deriv);

    gettimeofday(&conv, NULL);

    // direction and magnitude
    #pragma omp parallel for
    for(size_t i = 0; i < image.height * image.width; i++) {
        magnitude.data[i] = sqrt((hori.data[i] * hori.data[i]) + (vert.data[i] * vert.data[i]));
    }
    #pragma omp parallel for
    for(size_t i = 0; i < image.height * image.width; i++) {
        direction.data[i] = atan2(hori.data[i], vert.data[i]);
    }

    gettimeofday(&mag, NULL);

    suppression(&direction, &magnitude, &supp);

    gettimeofday(&sup, NULL);

    memcpy(temp.data, supp.data, sizeof(float) * supp.height * supp.width);
    mergeSort(temp.data, temp.width * temp.height, numthreads);

    float t_high = temp.data[(size_t) (temp.height * temp.width * 0.9)];
    float t_low = t_high / 5.0;
    
    gettimeofday(&sort, NULL);

    memcpy(temp.data, supp.data, sizeof(float) * supp.height * supp.width);
    hysteresis(&temp, t_high, t_low);

    gettimeofday(&doublethresh, NULL);

    edge_linking(&temp, &hyst);

    gettimeofday(&edge, NULL);

    // stop time
    gettimeofday(&compend, NULL);
    
    #pragma omp parallel sections
    {
        #pragma omp section
        write_image_template("direction.pgm", direction.data, direction.width, direction.height);
        #pragma omp section
        write_image_template("magnitude.pgm", magnitude.data, magnitude.width, magnitude.height);
        #pragma omp section
        write_image_template("suppression.pgm", supp.data, supp.width, supp.height);
        #pragma omp section
        write_image_template("hysteresis.pgm", hyst.data, hyst.width, hyst.height);
    }

    gettimeofday(&end, NULL);

    printf("%d, %.1f, %zu, %.1f, %.1f\n", 
        image.height, 
        sigma, 
        numthreads, 
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
            timecalc(doublethresh, edge), // edge_time
            timecalc(start, end) // total_time
        );
    }

    free(image.data);
    free(temp.data);
    free(vert.data);
    free(hori.data);
    free(magnitude.data);
    free(direction.data);
    free(hyst.data);
    free(supp.data);
    free(kern.data);
    return 0;
}

void img_prep(const img_s *orig, img_s *cpy) {
    cpy->height = orig->height;
    cpy->width = orig->width;
    cpy->data = (float *) calloc(cpy->height * cpy ->width, sizeof(float));
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
    #pragma omp parallel for
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] = exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum += kern->data[i];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] /= sum;
    }
}

void gaussian_deriv(kern_s *kern, float sigma, float a) {
    float sum = 0;

    kern->data = (float*) calloc(kern->w, sizeof(float));

    #pragma omp parallel for
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] = -1.0 * (i-a) * exp((-1.0 * (i-a) * (i-a)) / (2.0 * sigma * sigma));
        sum -= i * kern->data[i];
    }
    #pragma omp parallel for
    for (size_t i = 0; i < kern->w; i++) {
        kern->data[i] /= sum;
    }
    // kernel flipping
    #pragma omp parallel for
    for (size_t i = 0; i < (kern->w/2); i++) {
        float temp = kern->data[kern->w - 1 - i];
        kern->data[kern->w - 1 - i] = kern->data[i];
        kern->data[i] = temp;
    }
}

void h_conv(img_s *in_img, img_s *out_img, const kern_s *kern) {
    size_t bounds = in_img->width * in_img->height;
    int i_off = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < bounds; i++) {
        float sum = 0;
        for( size_t k = 0; k < kern->w; k++) {
            int offset = k - floor(kern->w / 2);
            i_off = i + offset;
            if (i_off / in_img->width == i / in_img->width) { // same row
                if (i_off < bounds && i_off >= 0) {
                    sum += in_img->data[i_off] * kern->data[k];
                }
            }
        }
        out_img->data[i] = sum;
    }
}


void v_conv(img_s *in_img, img_s *out_img, const kern_s *kern) {
    size_t bounds = in_img->width * in_img->height;
    int i_off = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < bounds; i++) {
        float sum = 0;
        for( size_t k = 0; k < kern->w; k++) {
            int offset = (k - floor(kern->w / 2)) * in_img->width;
            i_off = i + offset;
            if (i_off < bounds && i_off >= 0) {
                sum += in_img->data[i_off] * kern->data[k];
            }
        }
        out_img->data[i] = sum;
    }
}

void suppression(img_s *direction, img_s *magnitude, img_s *supp) {
    #define Gxy magnitude->data
    size_t bounds = direction->width * direction->height;
    size_t width = magnitude->width;
    size_t btm_right = width + 1;
    size_t btm_left = width - 1;
    float theta;
    #pragma omp parallel for
    for (size_t i = 0; i < bounds; i++) {
        theta = direction->data[i];
        if (theta < 0) {
            theta += M_PI;
        }
        theta *= (180.0 / M_PI);
        supp->data[i] = Gxy[i];
        if (theta <= 22.5 || theta > 157.5) {
            // top
            if (i >= width) {
                if (Gxy[i] < Gxy[i - width]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
            // bottom
            if (i < bounds - width) {
                if (Gxy[i] < Gxy[i + width]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
        } else if (theta > 22.5 && theta <= 67.5) {
            //topleft
            if (i >= width && i % width > 0) {
                if (Gxy[i] < Gxy[i - btm_right]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
            // bottomright
            if (i < bounds - width && i % width < width-1) {
                if (Gxy[i] < Gxy[i + btm_right]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
        } else if (theta > 67.5 && theta <= 112.5) {
            // left
            if (i % width > 0) {
                if (Gxy[i] < Gxy[i - 1]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
            // right
            if (i % width < width-1) {
                if (Gxy[i] < Gxy[i + 1]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
        } else if (theta > 112.5 && theta <= 157.5) {
            // topright
            if (i >= width && i % width < width-1) {
                if (Gxy[i] < Gxy[i - btm_left]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
            // bottomleft
            if (i < bounds - width && i % width > 0) {
                if (Gxy[i] < Gxy[i + btm_left]) {
                    supp->data[i] = 0;
                    continue;
                }
            }
        }
    }
    #undef Gxy
}

void hysteresis(img_s *img, float t_high, float t_low) {
    size_t bounds = img->height * img-> width;
    #pragma omp parallel for
    for(size_t i = 0; i < bounds; i++) {
        if (img->data[i] >= t_high) {
            img->data[i] = 255;
        } else if (img->data[i] <= t_low) {
            img->data[i] = 0;
        } else {
            img->data[i] = 125;
        }
    }
}

void edge_linking(img_s *hyst, img_s *edges) {
    size_t bounds = hyst->height * hyst-> width;
    size_t width = hyst->width;
    size_t btm_right = width + 1;
    size_t btm_left = width - 1;
    #pragma omp parallel for
    for (size_t i = 0 ; i < bounds; i++) {
        if(hyst->data[i] == 125) {
            // topleft
            if (i >= width && i % width > 0) {
                if (hyst->data[i - btm_right] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // top
            if (i >= width) {
                if (hyst->data[i - width] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // topright
            if (i >= width && i % width < width-1) {
                if (hyst->data[i - btm_left] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // left
            if (i % width > 0) {
                if (hyst->data[i - 1] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // right
            if (i % width > width-1) {
                if (hyst->data[i + 1] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // bottomleft
            if (i < bounds - width && i % width > 0) {
                if (hyst->data[i + btm_left] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // bottom
            if (i < bounds - width) {
                if (hyst->data[i + width] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            // bottomright
            if (i < bounds - width && i % width < width-1) {
                if (hyst->data[i + btm_right] == 255) {
                    edges->data[i] = 255;
                    continue;
                }
            }
            edges->data[i] = 0;
        } else {
            edges->data[i] = hyst->data[i];
        }
    }
}

float timecalc(struct timeval start, struct timeval end) {
    float ns = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    return ns / 1000.0;
}

int sortcomp(const void *e1, const void *e2) {
    float f1 = *(float *) e1;
    float f2 = *(float *) e2;
    return (f1 > f2) - (f1 < f2);
}
