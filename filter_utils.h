#ifndef FILTER_UTILS_H
#define FILTER_UTILS_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

enum FilterType {
    NONE,
    GRAYSCALE,
    HDR_TONEMAPPING,
    WIPE_TRANSITION
};

// pipeline 
struct FilterPipeline {
    std::vector<FilterType> active_filters;
    cudaStream_t stream; // cuda streams 
    float transition_t;  // wipe transition 0.0 - 1.0 
};

#endif