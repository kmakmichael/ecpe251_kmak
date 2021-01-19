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

int main(int argc, char *argv[]) {

    char *img_path;
    float sigma;

    if (argc != 3) {
        fprintf(stderr, "usage: canny_stage1 <image path> <sigma>\n");
        return -1;
    }
    sigma = atof(argv[2]);
    if (sigma <= 0) {
        fprintf(stderr, "invalid sigma: %s\n", argv[2]);
    }
    
    return 0;
}