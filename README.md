# 🚀 CUDA Webcam Filter Pipeline & Transitions

This project implements a high-performance, multi-stage **Filter Pipeline** using NVIDIA CUDA. It allows multiple image filters to be chained together and applied sequentially to a 1080p video stream while maintaining ultra-low latency.

## 📊 Performance Analysis (Part 3)
Benchmarks were conducted on an **NVIDIA GeForce GTX 1070** for a Full HD (**1920x1080**) frame.

| Configuration | Latency (ms) | FPS Capability | Status |
| :--- | :--- | :--- | :--- |
| Sequential CPU (Estimated) | ~280.00 ms | 3.5 FPS | Bottleneck |
| **CUDA Filter Pipeline** | **2.243 ms** | **445.8 FPS** | **Real-Time Ready** |

### Optimization Strategy
- **Intermediate Result Management:** By using dedicated GPU memory buffers (`d_temp`), we avoid transferring data back to the CPU between filter stages. This eliminates the PCIe bottleneck.
- **Asynchronous Execution:** The architecture utilizes **CUDA Streams** to manage execution, allowing for future extensions where memory copies and kernel executions can overlap.

---

## 🏗️ Pipeline Architecture (Part 1)
The system is designed with a **Sequential Chain Architecture**. Each filter in the pipeline reads from the previous stage's output buffer and writes to the next.

**Key Features:**
- **Dynamic Filter Chaining:** Filters can be added or removed from the `FilterPipeline` vector.
- **Stream-Based Processing:** All operations are assigned to a specific `cudaStream_t` to ensure non-blocking execution relative to the host.

---

## 🎭 Custom Transitions (Part 2)
We implemented a **Wipe Transition** kernel that allows for smooth visual switching between two states (e.g., Original Image vs. Filtered Image).

- **Left-to-Right Wipe:** Uses a normalized parameter $t \in [0.0, 1.0]$ to determine the boundary.
- **Dynamic Timing:** The transition works frame-by-frame, allowing for real-time adjustments of the "wipe" position.

---

## 🛠️ Implementation Details

### Filters Included:
1.  **Grayscale:** Converts 3-channel RGB data into a weighted luminance representation.
2.  **Wipe:** Performs a spatial blend between two input buffers based on a vertical boundary.

### Compilation & Running
Ensure you have the CUDA Toolkit installed. This test uses a synthetic 1080p stream for precise performance measurement.

```bash
# Compilation
nvcc main.cpp pipeline_kernels.cu -o pipeline_test
pipeline_test
