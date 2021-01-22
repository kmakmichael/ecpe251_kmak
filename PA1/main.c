/*
    Michael Kmak
    ECPE 251 - High-Performance Computing
    PA1 - Canny Edge Detector

    usage: ./canny_stage1 <image path> <sigma>

    quick description...
*/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"

#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"
#include "math.h"
#include "sys/time.h"

#include "image_template.h"

#pragma GCC diagnostic pop

// structs
typedef struct {
    int *data;
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
void convolution(img_s *image, const kern_s *h_kern, const kern_s *v_kern);
void img_copy(const img_s *orig, img_s *cpy);
void h_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void v_conv(img_s *in_img, img_s *out_img, const kern_s *kern);
void print_img(img_s *image);


int main(int argc, char *argv[]) {

    img_s image;
    img_s temp;
    img_s vert;
    img_s hori;
    img_s magnitude;
    img_s direction;
    float sigma;
    float a;
    kern_s kern;
    //kern_s v_kern;
    //kern_s h_deriv;
    //kern_s v_deriv;
    struct timeval start, end;

    if (argc != 3) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
    }

    read_image_template(argv[1], &image.data, &image.width, &image.height);
    img_copy(&image, &temp);
    img_copy(&image, &vert);
    img_copy(&image, &hori);
    img_copy(&image, &magnitude);
    img_copy(&image, &direction);

    // kernel initialization
    a = round(2.5 * sigma - 0.5);
    kern.w = 2 * a + 1;
    kern.data = (float*) calloc(kern.w, sizeof(float));

    // begin time
    gettimeofday(&start, NULL);

    // horizontal
    gaussian_kern(&kern, sigma, a);
    h_conv(&image, &temp, &kern);
    gaussian_deriv(&kern, sigma, a);
    h_conv(&temp, &hori, &kern);
    write_image_template("output/temp_hori.pgm", temp.data, temp.width, temp.height);
    write_image_template("output/hori.pgm", hori.data, hori.width, hori.height);

    // vertical
    gaussian_kern(&kern, sigma, a);
    v_conv(&image, &temp, &kern);
    gaussian_deriv(&kern, sigma, a);
    v_conv(&temp, &vert, &kern);
    write_image_template("output/temp_vert.pgm", temp.data, temp.width, temp.height);
    write_image_template("output/vert.pgm", vert.data, vert.width, vert.height);

    // direction and magnitude
    for(size_t i = 0; i < image.height * image.width; i++) {
        magnitude.data[i] = sqrt((hori.data[i] * hori.data[i]) + (vert.data[i] * vert.data[i]));
    }
    for(size_t i = 0; i < image.height * image.width; i++) {
        direction.data[i] = atan2(hori.data[i], vert.data[i]);
    }
    write_image_template("output/direction.pgm", direction.data, direction.width, direction.height);
    write_image_template("output/magnitude.pgm", magnitude.data, magnitude.width, magnitude.height);

    // stop time
    gettimeofday(&end, NULL);

    float runtime = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    printf("Runtime: %f ms\n", runtime / 1000.0);


    free(image.data);
    free(vert.data);
    free(hori.data);
    free(magnitude.data);
    free(direction.data);
    free(kern.data);
    return 0;
}

void img_copy(const img_s *orig, img_s *cpy) {
    cpy->height = orig->height;
    cpy->width = orig->width;
    cpy->data = (int *) calloc(cpy->height * cpy ->width, sizeof(int));
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
