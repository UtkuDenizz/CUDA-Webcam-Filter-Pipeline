#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "filter_utils.h"

extern "C" void apply_grayscale(unsigned char* d_in, unsigned char* d_out, int w, int h, cudaStream_t stream);
extern "C" void apply_wipe(unsigned char* d_a, unsigned char* d_b, unsigned char* d_out, int w, int h, float t, cudaStream_t stream);

int main() {
    int w = 1920, h = 1080;
    size_t size = w * h * 3;

    // memory management (intermediate results)
    unsigned char *d_input, *d_temp, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_output, size);

    // creating a stream (concurrent processing)
    FilterPipeline myPipe;
    myPipe.active_filters = {GRAYSCALE, WIPE_TRANSITION};
    myPipe.transition_t = 0.5f; // half of the screen
    cudaStreamCreate(&myPipe.stream);

    // performance analysis
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);

    // PIPELINE EXECUTION
    // grayscale (input -> temp)
    apply_grayscale(d_input, d_temp, w, h, myPipe.stream);
    
    // Wipe (combine input and grayscale -> Output)
    apply_wipe(d_input, d_temp, d_output, w, h, myPipe.transition_t, myPipe.stream);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Pipeline Performance (Full HD): " << ms << " ms" << std::endl;

    // cleaning
    cudaStreamDestroy(myPipe.stream);
    cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output);

    return 0;
}