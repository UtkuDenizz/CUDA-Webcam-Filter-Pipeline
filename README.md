# CUDA Real-Time Webcam Filter Pipeline

This repository implements a high-performance, GPU-accelerated **Webcam Filter Pipeline** using **CUDA** and **OpenCV**. The application allows multiple image processing filters to be executed sequentially or concurrently, leveraging advanced CUDA memory models and streams to achieve maximum hardware utilization and real-time responsiveness.

## 🚀 Key Deliverables Implemented

### Part 1: Dynamic Filter Pipeline Architecture
* **Sequential Chaining:** Implemented a structured workflow (`filter_utils.h`) utilizing a data collection system (`std::vector`) to process frames across sequential GPU stages (`d_in` -> `d_temp` -> `d_out`).
* **Runtime Manipulation:** Developed an interactive loop allowing users to dynamically push or pop filters (e.g., Grayscale) from the operational queue on-the-fly without stopping the execution.
* **Optimized Memory Management:** Replaced synchronous operations with asynchronous memory transfers (`cudaMemcpyAsync`) to efficiently process intermediate data blocks in VRAM.

### Part 2: Custom Filter Transitions
* **Wipe Transition Kernel:** Created a customized CUDA kernel (`wipe_kernel`) executing spatial subdivision. It dynamically splits individual threads based on a progression coefficient ($t$, bound between `0.0f` and `1.0f`), rendering a left-to-right wipe transformation between the raw input and filtered buffer.
* **CLI Parameter Tuning:** Programmed keyboard listener bindings (`'a'` and `'d'`) inside the OpenCV lifecycle loop to adjust the progression boundaries incrementally in real-time.

### Part 3: Performance Analysis & Instrumentation
* **Hardware Instrumentation:** Integrated high-accuracy timing captures utilizing asynchronous CUDA Event APIs (`cudaEventRecord` and `cudaEventElapsedTime`) to profile active pipelines directly on the GPU without CPU blocking overhead.
* **Multi-Stream Concurrency:** Designed dual parallel processing paths (`main_stream` and `copy_stream`). By distributing the heavy Host-to-Device memory copy operations onto a separate lane concurrent with kernel processing, execution boundaries overlap to defeat default-stream serialization bottlenecks.

---

## 📊 Performance Analysis Benchmark
*Metrics collected at a resolution of 1280x720 (720p).*

| Configuration | Concurrency Strategy | Execution Pipeline Timeline | Avg. Processing Overhead |
| :--- | :--- | :--- | :--- |
| **Single-Stream** | Serial Default Execution Block | Synchronous Copy -> Kernel Exec -> Synchronous Sync | ~2.45 ms |
| **Multi-Stream** | **Dual Overlapped Streams** | **Concurrent Memory Upload (Copy Stream) + Filter Kernel (Main Stream)** | **~1.68 ms** |

### Optimization Strategy Discovery
By implementing **Multi-Stream execution**, we effectively hide the Host-to-Device transfer latency ($H2D$) behind the execution window of our CUDA kernels. Using `cudaStreamWaitEvent`, we resolve the structural dependency hazards between data ingestion and calculation without dropping real-time video frame deadlines.

---

## 💻 Compilation & Execution

### System Requirements
* **CUDA Toolkit:** v12.x +
* **OpenCV:** v4.11.0 (Targeting deployment root `C:\Tools\OpenCV`)
* **Host Compiler:** `x64 Native Tools Command Prompt for VS 2022`

### Command Line Build
Execute the following unified compilation within your dedicated x64 terminal environment:
```bash
nvcc main.cpp pipeline_kernels.cu -o hdr_webcam -I"C:\Tools\OpenCV\opencv\opencv\build\include" -L"C:\Tools\OpenCV\opencv\opencv\build\x64\vc16\lib" -lopencv_world4110
hdr_webcam