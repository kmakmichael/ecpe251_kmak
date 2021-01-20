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

#include "image_template.h"

#pragma GCC diagnostic pop

// structs
typedef struct {
    uint8_t *data;
    int width;
    int height;
} img_s;

typedef struct {
    uint8_t *data;
    size_t w;
} kern_s;

// functions
void gaussian_kernel(kern_s *kern, float sigma);


int main(int argc, char *argv[]) {

    img_s image;
    float sigma;
    kern_s h_kern;
    kern_s v_kern;

    if (argc != 3) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
    }

    read_image_template(argv[1], &image.data, &image.width, &image.height);
    gaussian_kernel(&h_kern, sigma);
    gaussian_kernel(&v_kern, sigma);

    free(image.data);
    return 0;
}


void gaussian_kern(kern_s *kern, float sigma) {
    uint8_t a = 2.5 * sigma; // add 0.5 and truncate instead of rounding, same result
    kern->w = 2 * a + 1;
    uint8_t sum = 0;

    kern->data = (uint8_t*) calloc(kern->w, sizeof(uint8_t));

    for (size_t i = 0; i < (kern->w - 1); i++) {
        kern->data[i] = exp((-1 * (i-a) * (i-a) / (2 * sigma * sigma)));
        sum += kern->data[i];
    }
    for (size_t i = 0; i < (kern->w - 1); i++) {
        kern->data[i] /= sum;
    }
}
