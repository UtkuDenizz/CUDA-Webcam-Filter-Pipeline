#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>
#include "filter_utils.h"

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

    size_t size = width * height * 3;
    unsigned char *d_in, *d_temp, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_temp, size);
    cudaMalloc(&d_out, size);

    FilterPipeline pipeline;
    pipeline.multi_stream_mode = false;
    pipeline.transition_t = 0.5f;
    pipeline.last_execution_ms = 0.0f;
    pipeline.active_filters.push_back(GRAYSCALE); 

    cudaStreamCreate(&pipeline.main_stream);
    cudaStreamCreate(&pipeline.copy_stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cv::Mat frame(height, width, CV_8UC3);
    cv::Mat output_frame(height, width, CV_8UC3);

    std::cout << "=== control keys ===" << std::endl;
    std::cout << "g: toggle grayscale | m: toggle multi-stream" << std::endl;
    std::cout << "a: decrease wipe t  | d: increase wipe t" << std::endl;
    std::cout << "q: exit program" << std::endl;

    while (true) {
        if (is_camera) {
            cap >> frame;
            if (frame.empty()) break;
        } else {
            frame = cv::Scalar(60, 0, 0);
            static int pos = 0;
            cv::circle(frame, cv::Point(pos % width, height / 2), 90, cv::Scalar(0, 255, 255), -1);
            cv::putText(frame, "synthetic pipeline", cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(255, 255, 255), 2);
            pos += 12;
        }

        cudaEventRecord(start, pipeline.main_stream);

        cudaStream_t upload_stream = pipeline.multi_stream_mode ? pipeline.copy_stream : pipeline.main_stream;
        cudaMemcpyAsync(d_in, frame.data, size, cudaMemcpyHostToDevice, upload_stream);

        if (pipeline.multi_stream_mode) {
            cudaEvent_t copy_ready;
            cudaEventCreate(&copy_ready);
            cudaEventRecord(copy_ready, pipeline.copy_stream);
            cudaStreamWaitEvent(pipeline.main_stream, copy_ready, 0);
            cudaEventDestroy(copy_ready);
        }

        bool has_grayscale = false;
        for (size_t i = 0; i < pipeline.active_filters.size(); i++) {
            if (pipeline.active_filters[i] == GRAYSCALE) {
                has_grayscale = true;
            }
        }

        // workflow fix: always generate filtered version to d_temp first
        apply_grayscale(d_in, d_temp, width, height, pipeline.main_stream);

        if (has_grayscale) {
            // apply wipe transition between raw input (d_in) and filtered stream (d_temp)
            apply_wipe(d_in, d_temp, d_out, width, height, pipeline.transition_t, pipeline.main_stream);
        } else {
            // if grayscale is disabled, output raw frame directly without transition
            cudaMemcpyAsync(d_out, d_in, size, cudaMemcpyDeviceToDevice, pipeline.main_stream);
        }

        cudaMemcpyAsync(output_frame.data, d_out, size, cudaMemcpyDeviceToHost, pipeline.main_stream);
        cudaStreamSynchronize(pipeline.main_stream);

        cudaEventRecord(stop, pipeline.main_stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&pipeline.last_execution_ms, start, stop);

        std::string mode_str = pipeline.multi_stream_mode ? "multi-stream" : "single-stream";
        std::string metrics = "mode: " + mode_str + " | time: " + std::to_string(pipeline.last_execution_ms) + " ms | wipe: " + std::to_string(pipeline.transition_t);
        cv::putText(output_frame, metrics, cv::Point(30, height - 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow("cuda filter pipeline & wipe transition", output_frame);

        char key = (char)cv::waitKey(1);
        if (key == 'q') break;
        if (key == 'm') pipeline.multi_stream_mode = !pipeline.multi_stream_mode;
        if (key == 'a') pipeline.transition_t = std::max(0.0f, pipeline.transition_t - 0.05f);
        if (key == 'd') pipeline.transition_t = std::min(1.0f, pipeline.transition_t + 0.05f);
        if (key == 'g') {
            if (has_grayscale) {
                pipeline.active_filters.clear(); 
            } else {
                pipeline.active_filters.push_back(GRAYSCALE); 
            }
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(pipeline.main_stream);
    cudaStreamDestroy(pipeline.copy_stream);
    cudaFree(d_in);
    cudaFree(d_temp);
    cudaFree(d_out);
    return 0;
}