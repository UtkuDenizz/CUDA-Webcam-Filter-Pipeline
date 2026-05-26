#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "filter_utils.h"

// external calls to cuda kernel wrappers for the pipeline
extern "C" void apply_grayscale(unsigned char* d_in, unsigned char* d_out, int w, int h, cudaStream_t stream);
extern "C" void apply_wipe(unsigned char* d_a, unsigned char* d_b, unsigned char* d_out, int w, int h, float t, cudaStream_t stream);

int main() {
    cv::VideoCapture cap(0);
    int width = 1280, height = 720;
    bool is_camera = cap.isOpened();

    if (is_camera) {
        width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    }

    // memory allocation for pipeline (input -> intermediate -> output)
    size_t size = width * height * 3;
    unsigned char *d_in, *d_temp, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_out, size);

    // stream creation 
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat output_frame(height, width, CV_8UC3);

    std::cout << (is_camera ? "Status: Camera Active" : "Status: Synthetic Mode (Pipeline Test)") << std::endl;

    while (true) {
        if (is_camera) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            // synthetic stream generator
            frame = cv::Scalar(60, 0, 0); // deep blue background
            static int pos = 0;
            // draw a moving yellow circle to simulate dynamic input
            cv::circle(frame, cv::Point(pos % width, height / 2), 90, cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, "SYNTHETIC PIPELINE", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
            pos += 12;
        }
        cudaMemcpyAsync(d_in, frame.data, size, cudaMemcpyHostToDevice, stream);

        // grayscale conversion
        apply_grayscale(d_in, d_temp, width, height, stream);
        
        // wipe transition (Comparing original vs filtered)
        // half screen
        apply_wipe(d_in, d_temp, d_out, width, height, 0.5f, stream); 

        cudaMemcpyAsync(output_frame.data, d_out, size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cv::imshow("CUDA Filter Pipeline & Wipe Transition", output_frame);
        if (cv::waitKey(1) == 'q') break;
    }

    // cleanup 
    cudaStreamDestroy(stream);
    cudaFree(d_in); 
    cudaFree(d_temp); 
    cudaFree(d_out);
    return 0;
}