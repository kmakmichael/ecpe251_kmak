/*
    Michael Kmak
    ECPE 251 - High-Performance Computing
    PA1 - Canny Edge Detector

    usage: ./canny_stage1 <image path> <sigma>

    quick description...
*/

#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "math.h"
#include "sys/time.h"
#include "omp.h"

#include "image_template.h"

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
void img_prep(const img_s *orig, img_s *cpy);
void h_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void v_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void suppression(img_s *direction, img_s *magnitude, img_s *out_img);
void hyst(img_s *img, float t_high, float t_low);
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
    size_t numthreads;
    kern_s kern;
    struct timeval readstart, compstart, conv, mag, sup, sort, doublethresh, edge, end;
    float sigma;
    float a;

    if (argc != 4) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma> <# threads>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
    }
    numthreads = atoi(argv[3]);
    if (numthreads <= 0) {
        fprintf(stderr, "invalid number of threads: %s\n", argv[3]);
    }
    omp_set_num_threads(numthreads);

    // begin time
    gettimeofday(&readstart, NULL);
    read_image_template(argv[1], &image.data, &image.width, &image.height);
    img_prep(&image, &temp);
    img_prep(&image, &vert);
    img_prep(&image, &hori);
    img_prep(&image, &magnitude);
    img_prep(&image, &direction);

    // kernel initialization
    a = round(2.5 * sigma - 0.5);
    kern.w = 2 * a + 1;
    kern.data = (float*) calloc(kern.w, sizeof(float));

    // begin time
    gettimeofday(&compstart, NULL);

    // horizontal
    gaussian_kern(&kern, sigma, a);
    h_conv(&image, &temp, &kern);
    gaussian_deriv(&kern, sigma, a);
    h_conv(&temp, &hori, &kern);

    // vertical
    gaussian_kern(&kern, sigma, a);
    v_conv(&image, &temp, &kern);
    gaussian_deriv(&kern, sigma, a);
    v_conv(&temp, &vert, &kern);

    // direction and magnitude
    for(size_t i = 0; i < image.height * image.width; i++) {
        magnitude.data[i] = sqrt((hori.data[i] * hori.data[i]) + (vert.data[i] * vert.data[i]));
    }
    for(size_t i = 0; i < image.height * image.width; i++) {
        direction.data[i] = atan2(hori.data[i], vert.data[i]);
    }
    write_image_template("direction.pgm", direction.data, direction.width, direction.height);
    write_image_template("magnitude.pgm", magnitude.data, magnitude.width, magnitude.height);

    // re-use vert as suppression and hori as hysteresis to save some memory so my VM can handle the bigger images
    suppression(&direction, &magnitude, &vert);
    write_image_template("suppression.pgm", vert.data, vert.width, vert.height);

    memcpy(temp.data, vert.data, sizeof(float) * vert.height * vert.width);
    qsort(temp.data, temp.width * temp.height, sizeof(float), sortcomp);
    float t_high = temp.data[(size_t) (temp.height * temp.width * 0.9)];
    float t_low = t_high / 5.0;
    hyst(&vert, t_high, t_low);
    edge_linking(&vert, &hori);
    write_image_template("hysteresis.pgm", hori.data, hori.width, hori.height);

    // stop time
    gettimeofday(&end, NULL);

    printf("%d, %.1f, %.1f, %.1f\n", image.height, sigma, timecalc(compstart, end), timecalc(readstart, end));

    free(image.data);
    free(temp.data);
    free(vert.data);
    free(hori.data);
    free(magnitude.data);
    free(direction.data);
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

void h_conv(img_s *in_img, img_s *out_img, const kern_s *kern) {
    size_t bounds = in_img->width * in_img->height;
    int i_off = 0;
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

void hyst(img_s *img, float t_high, float t_low) {
    size_t bounds = img->height * img-> width;
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
