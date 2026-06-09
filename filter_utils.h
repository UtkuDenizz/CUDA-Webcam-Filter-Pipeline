#ifndef FILTER_UTILS_H
#define FILTER_UTILS_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

enum FilterType {
    NONE,
    GRAYSCALE,
    HDR_TONEMAPPING, // keep placeholder for curricular requirement
    WIPE_TRANSITION
};

// pipeline architecture to manage runtime configuration
struct FilterPipeline {
    std::vector<FilterType> active_filters;
    cudaStream_t main_stream;
    cudaStream_t copy_stream; // secondary stream for concurrent operations
    bool multi_stream_mode;   // toggle flag for performance comparison
    float transition_t;       // wipe progress factor (0.0f - 1.0f)
    float last_execution_ms;  // instrumentation storage for real-time tracking
};

#endif