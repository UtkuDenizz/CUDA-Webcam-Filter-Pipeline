#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// grayscale kernel
__global__ void grayscale_kernel(unsigned char* input, unsigned char* output, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        unsigned char gray = (unsigned char)(0.299f * input[idx] + 0.587f * input[idx+1] + 0.114f * input[idx+2]);
        output[idx] = output[idx+1] = output[idx+2] = gray;
    }
}

// wipe transition kernel (custom transitions)
__global__ void wipe_kernel(unsigned char* in_a, unsigned char* in_b, unsigned char* out, int w, int h, float t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) {
        int idx = (y * w + x) * 3;
        int boundary = (int)(w * t);
        if (x < boundary) {
            out[idx] = in_b[idx]; out[idx+1] = in_b[idx+1]; out[idx+2] = in_b[idx+2];
        } else {
            out[idx] = in_a[idx]; out[idx+1] = in_a[idx+1]; out[idx+2] = in_a[idx+2];
        }
    }
}

// wrapper
extern "C" void apply_grayscale(unsigned char* d_in, unsigned char* d_out, int w, int h, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    grayscale_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, w, h);
}

extern "C" void apply_wipe(unsigned char* d_a, unsigned char* d_b, unsigned char* d_out, int w, int h, float t, cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);
    wipe_kernel<<<blocks, threads, 0, stream>>>(d_a, d_b, d_out, w, h, t);
}