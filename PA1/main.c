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

#include "image_template.h"

#pragma GCC diagnostic pop

typedef struct {
    uint8_t **data;
    int width;
    int height;
} img_s;

int main(int argc, char *argv[]) {

    img_s image;
    float sigma;

    if (argc != 3) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
    }

    read_image_template(argv[1], &image.data, &image.width, &image.height);
    
    return 0;
}