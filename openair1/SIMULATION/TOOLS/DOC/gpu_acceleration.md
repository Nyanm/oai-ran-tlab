# GPU-Accelerated Channel Simulation

## 1\. Overview

This document describes the CUDA-based GPU acceleration pipeline for the OAI channel simulation. The primary goal of this feature is to offload the computationally intensive `multipath_channel` convolution and `add_noise` functions from the CPU to the GPU. This overcomes existing performance bottlenecks and enables large-scale, real-time physical layer simulation that is not feasible with the CPU-based models.

The result is a complete, high-performance pipeline integrated into the OAI simulators. The feature is supported by a new benchmark suite created to validate correctness and analyze performance across a wide range of channel models and hardware configurations.

## 2\. Implementation

This feature is implemented across a set of new CUDA source files and integrated into the existing OAI simulation framework.

### 2.1 Multipath Channel Convolution (`multipath_channel.cu`)

The core of the channel simulation is the multipath convolution. On the CPU, this is a sequential process where each output sample is calculated by iterating through all channel taps and accumulating the results. To parallelize this on the GPU, we assign one CUDA thread to calculate one output sample.

The key to making this parallel approach efficient is the use of a tiled convolution pattern with `__shared__` memory. Instead of having every thread read from slow global memory for each step of the convolution, threads in a block cooperate to pre-fetch all the data they will need into the on-chip shared memory.

To further optimize this, the responsibility for handling boundary conditions (the "halo" of past samples) has been moved from the GPU kernel to the host. The host code now prepares a larger input buffer that is **pre-padded** with `channel_length - 1` zeros before the actual signal data. This allows the kernel to be significantly simplified by removing the previous boundary-checking logic.

```cpp
// Snippet from the multipath_channel_kernel in multipath_channel.cu

// Step 1: Each thread cooperates to load a tile of the pre-padded signal
// into fast shared memory.
const int padding_len = channel_length - 1;
const int padded_num_samples = num_samples + padding_len;

for (int k = tid; k < shared_mem_size; k += blockDim.x) {
    int load_idx = block_start_idx + k;
    // Unconditional load from the padded buffer
    int interleaved_idx = 2 * (j * padded_num_samples + load_idx);
    tx_shared[k] = make_float2(tx_sig[interleaved_idx], tx_sig[interleaved_idx + 1]);
}
__syncthreads();
```

Once the data is staged in fast shared memory, each thread can perform the full convolution for its assigned output sample. The inner loop iterates through the `channel_length` taps, and all reads of the input signal (`tx_sample`) come from the `tx_shared` buffer.

```cpp
// Snippet from multipath_channel_kernel in multipath_channel.cu
// Step 2: Each thread performs convolution using the data in shared memory
    for (int l = 0; l < channel_length; l++) {
        // Read the historical transmit sample from the shared memory
        float2 tx_sample = tx_shared[tid + (channel_length - 1) - l];
        
        // Get the corresponding channel weight from global memory
        int chan_link_idx = ii + (j * nb_rx);
        float2 chan_weight = d_channel_coeffs[chan_link_idx * channel_length + l];
        
        // Perform the complex multiply-accumulate
        rx_tmp = complex_add(rx_tmp, complex_mul(tx_sample, chan_weight));
    }
    __syncthreads();
}

// Write the final result from the thread's private register to global memory...
```

### 2.2 Noise Generation (`phase_noise.cu`)

After the multipath convolution, the next stage of the pipeline is to add realistic noise to the signal. This includes both Additive White Gaussian Noise (AWGN) and phase noise.

The CPU implementation performs this in a simple loop. For each sample of the signal, it calls a standard C library function to generate pseudo-random numbers, scales them appropriately, and adds them to the sample. This process is sequential and computationally inexpensive for a single signal but becomes a bottleneck when simulating many channels that all require high-quality random numbers simultaneously.

The GPU port follows the same parallelization strategy as the multipath kernel: one CUDA thread is assigned to process one single sample. The main challenge is generating millions of high-quality random numbers in parallel. To achieve this, the implementation relies on NVIDIA's **`cuRAND`** library.

The `cuRAND` library is stateful, meaning each thread needs its own unique generator "state" to produce an independent sequence of random numbers. In our implementation, a pool of `curandState_t` objects is initialized once in global memory. When the noise kernel is launched, each thread is assigned a unique state from this pool. The thread uses its state to generate random numbers, and then writes the updated state back to global memory for the next use.

A simplified view of this logic within the `add_noise_and_phase_noise_kernel` is shown below:

```cpp
// Simplified snippet from add_noise_and_phase_noise_kernel in phase_noise.cu
__global__ void add_noise_and_phase_noise_kernel(...)
{
    // Each thread calculates its unique sample index 'i' and antenna index 'ii'...

    // Load this thread's unique random number generator state
    curandState_t local_state = states[ii * num_samples + i];
    
    // Generate random numbers using the local state
    // Generates two normal-distributed numbers for AWGN (real and imaginary)
    float2 awgn = curand_normal2(&local_state);
    //    Generates one normal-distributed number for phase noise
    float phase_error = curand_normal(&local_state) * pn_std_dev;

    // Apply noise to the input signal from the previous kernel stage, and then aplu phase_error

    // Write the updated state back to global memory for the next simulation run
    states[ii * num_samples + i]= local_state; 

    // Clamp the final floating-point value and store it as a short integer
    output_sig[ii * num_samples + i]= make_short2(...);
}
```

### 2.3 Pipeline Orchestration (`channel_pipeline.cu`)

The `channel_pipeline.cu` file serves two primary functions: providing an interoperable API and orchestrating kernel execution.

Its first purpose is to act as an abstraction layer between the host C/C++ code of the OAI simulator and the CUDA C++ device code. It exposes a C-style Application Programming Interface (API) using `extern "C"`, which allows the standard C-based components to call the GPU pipeline without needing to handle CUDA-specific syntax or complexities.

Its second purpose is to orchestrate the execution of the individual CUDA kernels. It defines the logical flow of the channel simulation by launching the `multipath_channel_kernel` and `add_noise_and_phase_noise_kernel` in sequence and managing the data dependencies between them.

The functions within this file, such as `run_channel_pipeline_cuda`, implement a consistent workflow. Upon being called, they first cast the generic `void*` pointers received from the C host into their specific CUDA data types. The implementation then contains the conditional logic, via preprocessor directives, to support the various memory models. Following data setup, the functions define the CUDA execution grid and block dimensions and launch the two kernels sequentially. The output of the multipath kernel is directed to an intermediate buffer, which serves as the input for the subsequent noise kernel. Finally, the functions manage synchronization with the host and handle the transfer of the final results from the GPU back to host memory.

*The external API of these pipeline functions has been streamlined. The function signatures no longer require a separate `tx_sig_interleaved` parameter, instead relying on a single, pre-padded host buffer (`h_tx_sig_pinned`) as the authoritative source for the input signal, which simplifies the data flow from the simulators.*

---

## 3\. Memory Management Models

To provide flexibility and enable performance testing on different hardware architectures, the project now supports three distinct memory management models. The desired model is selected at build time by passing a flag to the CMake build command.

### 3.1 ATS Hybrid Model (`-DUSE_ATS_MEMORY=ON`)

This is a hybrid approach that leverages Address Translation Services (ATS), a hardware feature allowing the GPU to directly access host-allocated memory. In our implementation, the large input signal buffer is allocated in host memory using `malloc`. The GPU kernel is then able to read this data directly from the host, eliminating the need for an initial Host-to-Device `cudaMemcpy` for that buffer. Intermediate and output buffers are still allocated on the device, and the final result is copied back to the host explicitly. Due to its better performance compared to the other methods, it's also the default behaviour when no flag is specified.

### 3.2 Unified Memory Model (`-DUSE_UNIFIED_MEMORY=ON`)

This model simplifies memory management by creating a single, managed memory space that is accessible to both the CPU and the GPU. Memory is allocated once using `cudaMallocManaged`. The CUDA runtime and driver then automatically handle the migration of data to whichever processor is accessing it (e.g., moving data to the GPU's memory when a kernel is launched). This removes the need for most explicit `cudaMemcpy` calls. While the CUDA driver automatically migrates data on-demand, the implementation includes further optimizations to guide this process for better performance. `cudaMemAdvise` is used to provide performance hints to the driver, such as setting a buffer's preferred location to the GPU or marking it as mostly read-only. Additionally, `cudaMemPrefetchAsync` is also used to explicitly move data to the GPU's memory ahead of time. This ensures the data is already local when a kernel begins execution, which helps to hide the latency of the data migration.

### 3.3 Explicit Copy Model

This is the traditional CUDA programming model. The host and device have separate and distinct memory spaces. The programmer is responsible for all data management. Memory must be allocated on the device using `cudaMalloc`, and data must be manually transferred between the host and device using explicit `cudaMemcpy` calls before and after a kernel launch. This configuration can be run with other flags turned off by using `-DUSE_ATS_MEMORY=OFF` and `-DUSE_UNIFIED_MEMORY=OFF`.

-----

Excellent. Let's refine the **Project Integration Section**.

This section is already in good shape, but we need to update the description of the CPU and GPU paths to reflect the new, fairer benchmarking methodology we implemented. I've integrated the new logic into the existing text.

***

## 4. Project Integration

To ensure the feature is modular, the integration into the OAI project was handled at both the build system and source code levels. All of the new CUDA source files (`channel_pipeline.cu`, `multipath_channel.cu`, `phase_noise.cu`) are compiled into a single static library named `oai_cuda_lib`. This is defined in the main `CMakeLists.txt` file and is controlled by the `CUDA_ENABLE` CMake option, which is disabled by default. When this option is activated (`-DCUDA_ENABLE=ON`), the build system compiles the CUDA library and also defines a global `ENABLE_CUDA` preprocessor macro that is visible to the rest of the project.

This `ENABLE_CUDA` macro is then used within the simulator files, such as `nr_dlsim.c` and `nr_ulsim.c`, to conditionally compile all CUDA-related code. This approach allows for the inclusion of the `oai_cuda.h` header, the addition of the `--cuda` runtime flag, and the pre-allocation of GPU memory at startup, all without affecting the standard CPU-only build.

Inside the main processing loop, an `if (use_cuda)` statement acts as the primary runtime switch. To enable a direct comparison, timing measurement calls have been added to both the GPU and CPU execution paths. Also, both paths now source their data from the same common, pre-padded host buffer. The GPU path receives a pointer to the entire padded buffer and leverages its optimized kernel, while the CPU path uses a temporary array of offset pointers to work on the same data without being aware of the padding. This ensures both computations operate on the exact same source data, isolating the performance measurement to the channel processing itself. To execute the simulation on the GPU, the `--cuda` flag must be provided at runtime; otherwise, the program defaults to this refined CPU implementation.

Finally, the build system links the `nr_dlsim` and `nr_ulsim` executables against the `oai_cuda_lib` only when the feature is enabled, creating a clean separation between the two codebases.

---

## 5\. Benchmark and Analysis Suite

To validate the correctness, measure performance, and analyze the scalability of the GPU pipeline, a dedicated suite of tests was developed. The source code for these benchmarks is located in the `openair1/PHY/TOOLS/tests/` directory. This suite includes both focused unitary tests for individual components and a comprehensive benchmark for the end-to-end pipeline, which is the primary tool for performance evaluation.

### 5.1 `test_channel_scalability.c`

This program is the main tool for evaluating the end-to-end GPU channel simulation pipeline. It's designed to be highly configurable, allowing for the simulation of a wide variety of workloads and execution strategies.

#### 5.1.1 Configuration and Usage

The benchmark is configured at runtime using a set of command-line flags.

| Flag | Long Version | Argument | Description |
| :--- | :--- | :--- | :--- |
| `-c` | `--num-channels` | `<N>` | Sets the number of independent channels to simulate. |
| `-t` | `--nb-tx` | `<N>` | Sets the number of transmit antennas. |
| `-r` | `--nb-rx` | `<N>` | Sets the number of receive antennas. |
| `-s` | `--num-samples` | `<N>` | Sets the number of samples per signal. |
| `-l` | `--ch-len` | `<N>` | Sets the length of the channel impulse response. |
| `-n` | `--trials` | `<N>` | Sets the number of times each test is run for averaging. |
| `-S` | `--sum-outputs` | (none) | Enables interference simulation by summing channel outputs. (Using unique input signals) |
| `-m` | `--mode` | `<mode>` | Sets the GPU execution mode: `serial`, `stream`, or `batch`. |
| `-h` | `--help` | (none) | Displays the help message with all options. |

#### 5.1.2 Execution Modes

The `-m` or `--mode` flag controls the strategy used to execute the pipeline for multiple channels:

  * **`serial`**: This mode processes each channel sequentially. It loops through all channels and makes a synchronous call to `run_channel_pipeline_cuda` for each one, waiting for it to complete before starting the next. 
  * **`stream`**: This mode is designed to test concurrency. It launches all channel simulations asynchronously into separate CUDA streams by calling `run_channel_pipeline_cuda_streamed` in a loop. This allows the GPU hardware to overlap the execution of many kernels. 
  * **`batch`**: It aggregates the data for all channels on the host and makes a single call to `run_channel_pipeline_cuda_batched`. This launches one massive, unified kernel to process all channels at once, minimizing CPU overhead and maximizing GPU throughput. 

#### 5.1.3 Automated Analysis Script (`run_scalability_analysis.sh`)

To automate performance analysis, the `run_scalability_analysis.sh` script provides a convenient wrapper for the benchmark. It passes all command-line arguments directly to the `test_channel_scalability` executable while concurrently running `nvidia-smi` in the background to monitor the GPU's status, including utilization, power draw, and temperature. Upon completion, the script parses both the benchmark results and the GPU metrics to produce a single, consolidated report that correlates the pipeline's performance with the hardware's state during the test.

### 5.2 `test_channel_simulation.c`

The `test_channel_simulation.c` program was one of the initial benchmarks created for this project. Its primary function is to perform a direct performance comparison between the full, sequential CPU-based channel pipeline and the original synchronous GPU-based pipeline implemented in `run_channel_pipeline_cuda`. It served as an early-stage tool to validate the end-to-end functionality of the basic GPU port and to obtain initial speedup measurements before more advanced optimizations were developed.

### 5.3 `test_multipath.c`

The `test_multipath.c` program is a unitary test designed specifically to validate the `multipath_channel_cuda` function in isolation. The key feature of this test is the `verify_results` function, which performs a sample-by-sample comparison of the output arrays from both versions and calculates the Mean Squared Error (MSE). If the error is below a predefined tolerance, the test is marked as *"PASSED"* in the final report, confirming that the GPU kernel is producing numerically correct results.

### 5.4 `test_noise.c`

Similar to the multipath test, `test_noise.c` is a unitary test that focuses on a single component: the `add_noise_cuda` function. Its purpose is to validate the correctness of the GPU-based noise generation and measure its performance against the equivalent CPU version.

### 5.5 `test_SSE.c`

The `test_SSE.c` program is a CPU-focused benchmark designed to measure the performance impact of using Streaming SIMD Extensions (SSE) for the channel simulation. Its purpose is to compare the runtime of the standard, portable C implementation against a version that has been optimized with low-level SIMD intrinsics to perform parallel operations on the CPU.

To facilitate this comparison, the `CMakeLists.txt` file compiles the same `test_SSE.c` source file into two separate executables: `test_cpu_std` and `test_cpu_sse`. The `test_cpu_sse` version is compiled with the `CHANNEL_SSE` flag, which activates the optimized code paths within the channel simulation functions. The `test_cpu_std` version is compiled without this flag and therefore runs the standard C code. By running both executables with the same input and comparing their execution times, the performance gain from the SSE optimization can be measured.

-----

## 6\. Performance Analysis with the scalability test

The following results were generated by running the benchmark suites on a GH200 server. The data presented in this section was collected using different configurations and memory models.

### 6.1 **Explicit Copy**

#### **Test suite 1: Scalability vs. Channel Count**

This test measures performance by varying the number of channels for each of the three processing modes.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 7198.12  | 449.88  | 7  | 1.12  |
| 16 Channels, stream | 3785.73  | 236.61  | 9  | 2.12  |
| 16 Channels, batch | 5265.72  | 329.11  | 9  | 2.47  |
| 256 Channels, serial | 132292.92  | 516.77  | 18  | 12.18  |
| 256 Channels, stream | 65974.43  | 257.71  | 34  | 17.25  |
| 256 Channels, batch | 85696.71  | 334.75  | 23  | 12.59  |
| 1024 Channels, serial | 535187.04  | 522.64  | 19  | 16.45  |
| 1024 Channels, stream | 279650.69  | 273.10  | 38  | 27.63  |
| 1024 Channels, batch | 337217.10  | 329.31  | 36  | 17.90  |

-----

#### **Test suite 2: Performance vs. Channel Complexity**

This test measures how performance is affected by increasing the channel length (`-l`) for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 520979.26  | 508.77  | 36  | 14.54  |
| 16 Length, stream | 285613.37  | 278.92  | 30  | 23.02  |
| 16 Length, batch | 319082.56  | 311.60  | 29  | 14.78  |
| 64 Length, serial | 572750.90  | 559.33  | 29  | 21.18  |
| 64 Length, stream | 287145.10  | 280.42  | 47  | 36.90  |
| 64 Length, batch | 366564.52  | 357.97  | 51  | 23.06  |
| 128 Length, serial | 645907.17  | 630.77  | 51  | 28.65  |
| 128 Length, stream | 287374.70  | 280.64  | 75  | 55.03  |
| 128 Length, batch | 425883.51  | 415.90  | 82  | 32.18  |

-----

#### **Test suite 3: Performance vs. MIMO Configuration**

This test evaluates performance by changing the MIMO (e.g., `-t 2 -r 2`) configuration for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 263767.19  | 257.59  | 82  | 16.35  |
| 2x2 MIMO, stream | 117227.02  | 114.48  | 40  | 25.93  |
| 2x2 MIMO, batch | 158999.06  | 155.27  | 31  | 13.66  |
| 4x4 MIMO, serial | 540240.12  | 527.58  | 19  | 16.38  |
| 4x4 MIMO, stream | 292946.90  | 286.08  | 35  | 26.98  |
| 4x4 MIMO, batch | 336377.55  | 328.49  | 36  | 17.72  |
| 8x8 MIMO, serial | 1131519.91  | 1105.00  | 24  | 20.76  |
| 8x8 MIMO, stream | 614970.47  | 600.56  | 42  | 34.67  |
| 8x8 MIMO, batch | 718312.46  | 701.48  | 100  | 24.94  |

-----

#### **Test suite 4: Performance vs. Signal Samples**

This test assesses performance as the number of signal samples (`-s`) is varied for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 132058.81  | 128.96  | 30  | 19.09  |
| 30720 Samples, stream | 65519.89  | 63.98  | 48  | 26.67  |
| 30720 Samples, batch | 84702.49  | 82.72  | 50  | 14.35  |
| 61440 Samples, serial | 272354.08  | 265.97  | 21  | 17.49  |
| 61440 Samples, stream | 120543.25  | 117.72  | 47  | 32.09  |
| 61440 Samples, batch | 168761.94  | 164.81  | 40  | 16.36  |
| 122880 Samples, serial | 530901.68  | 518.46  | 31  | 17.22  |
| 122880 Samples, stream | 292913.50  | 286.05  | 34  | 26.60  |
| 122880 Samples, batch | 336500.40  | 328.61  | 36  | 17.68  |

---

### 6.2 **ATS hybrid Memory Model**

#### **Test Suite 1: Scalability vs. Channel Count**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 15607.20  | 975.45  | 4  | 1.20  |
| 16 Channels, stream | 9675.05  | 604.69  | 5  | 1.89  |
| 16 Channels, batch | 11446.41  | 715.40  | 4  | 1.56  |
| 256 Channels, serial | 254909.52  | 995.74  | 8  | 6.71  |
| 256 Channels, stream | 158545.91  | 619.32  | 11  | 8.85  |
| 256 Channels, batch | 186588.04  | 728.86  | 42  | 8.19  |
| 1024 Channels, serial | 1040642.57  | 1016.25  | 8  | 7.54  |
| 1024 Channels, stream | 641215.67  | 626.19  | 12  | 10.83  |
| 1024 Channels, batch | 716405.08  | 699.61  | 85  | 9.89  |

---

#### **Test Suite 2: Performance vs. Channel Complexity**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 1019507.99  | 995.61  | 30  | 6.38  |
| 16 Length, stream | 632961.15  | 618.13  | 10  | 9.06  |
| 16 Length, batch | 703812.27  | 687.32  | 64  | 7.55  |
| 64 Length, serial | 1058174.00  | 1033.37  | 23  | 10.65  |
| 64 Length, stream | 660569.55  | 645.09  | 17  | 15.19  |
| 64 Length, batch | 773648.96  | 755.52  | 98  | 13.50  |
| 128 Length, serial | 1155569.23  | 1128.49  | 45  | 15.39  |
| 128 Length, stream | 706371.55  | 689.82  | 26  | 23.39  |
| 128 Length, batch | 839650.26  | 819.97  | 100  | 19.31  |

---

#### **Test Suite 3: Performance vs. MIMO Configuration**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 472904.32  | 461.82  | 68  | 8.30  |
| 2x2 MIMO, stream | 235142.34  | 229.63  | 16  | 12.25  |
| 2x2 MIMO, batch | 341096.03  | 333.10  | 21  | 6.61  |
| 4x4 MIMO, serial | 1026054.51  | 1002.01  | 11  | 7.68  |
| 4x4 MIMO, stream | 636248.15  | 621.34  | 13  | 10.94  |
| 4x4 MIMO, batch | 733278.60  | 716.09  | 100  | 10.26  |
| 8x8 MIMO, serial | 2134596.16  | 2084.57  | 30  | 10.64  |
| 8x8 MIMO, stream | 1425572.90  | 1392.16  | 15  | 14.03  |
| 8x8 MIMO, batch | 1503629.66  | 1468.39  | 100  | 14.24  |

---

#### **Test Suite 4: Performance vs. Signal Samples**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 204935.85  | 200.13  | 46  | 13.31  |
| 30720 Samples, stream | 128429.91  | 125.42  | 23  | 16.07  |
| 30720 Samples, batch | 167390.00  | 163.47  | 37  | 9.31  |
| 61440 Samples, serial | 520550.37  | 508.35  | 10  | 8.32  |
| 61440 Samples, stream | 240315.01  | 234.68  | 20  | 15.46  |
| 61440 Samples, batch | 354902.66  | 346.58  | 28  | 8.60  |
| 122880 Samples, serial | 1020831.73  | 996.91  | 15  | 7.71  |
| 122880 Samples, stream | 647435.44  | 632.26  | 12  | 10.83  |
| 122880 Samples, batch | 730819.06  | 713.69  | 70  | 9.47  |


### 6.3 **Unified Memory Model**

#### **Test suite 1: Scalability vs. Channel Count**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 27008.22  | 1688.01  | 46  | 13.82  |
| 16 Channels, stream | 10653.77  | 665.86  | 32  | 8.78  |
| 16 Channels, batch | 16316.02  | 1019.75  | 35  | 10.32  |
| 256 Channels, serial | 422357.50  | 1649.83  | 71  | 58.09  |
| 256 Channels, stream | 170125.96  | 664.55  | 67  | 14.69  |
| 256 Channels, batch | 252856.97  | 987.72  | 92  | 48.37  |
| 1024 Channels, serial | 1737980.19  | 1697.25  | 73  | 64.95  |
| 1024 Channels, stream | 676768.18  | 660.91  | 69  | 15.81  |
| 1024 Channels, batch | 1044303.05  | 1019.83  | 100  | 59.01  |

***

#### **Test suite 2: Performance vs. Channel Complexity**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 1642949.82  | 1604.44  | 72  | 64.13  |
| 16 Length, stream | 674339.27  | 658.53  | 50  | 14.02  |
| 16 Length, batch | 973788.45  | 950.97  | 100  | 58.07  |
| 64 Length, serial | 1659572.32  | 1620.68  | 82  | 62.96  |
| 64 Length, stream | 695135.74  | 678.84  | 63  | 19.49  |
| 64 Length, batch | 1038129.00  | 1013.80  | 100  | 58.47  |
| 128 Length, serial | 1693933.59  | 1654.23  | 71  | 64.95  |
| 128 Length, stream | 736081.62  | 718.83  | 53  | 26.60  |
| 128 Length, batch | 992917.64  | 969.65  | 100  | 57.18  |

***

#### **Test suite 3: Performance vs. MIMO Configuration**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 641669.37  | 626.63  | 38  | 29.95  |
| 2x2 MIMO, stream | 255820.54  | 249.82  | 42  | 14.31  |
| 2x2 MIMO, batch | 529294.84  | 516.89  | 100  | 56.29  |
| 4x4 MIMO, serial | 1643607.18  | 1605.09  | 94  | 64.52  |
| 4x4 MIMO, stream | 681076.37  | 665.11  | 36  | 15.69  |
| 4x4 MIMO, batch | 1007215.55  | 983.61  | 100  | 58.65  |
| 8x8 MIMO, serial | 3250530.99  | 3174.35  | 93  | 64.34  |
| 8x8 MIMO, stream | 1696867.15  | 1657.10  | 74  | 45.36  |
| 8x8 MIMO, batch | 2104413.01  | 2055.09  | 100  | 59.99  |

***

#### **Test suite 4: Performance vs. Signal Samples**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 310191.51  | 302.92  | 45  | 34.16  |
| 30720 Samples, stream | 138705.06  | 135.45  | 40  | 17.83  |
| 30720 Samples, batch | 330750.69  | 323.00  | 100  | 58.12  |
| 61440 Samples, serial | 663344.93  | 647.80  | 72  | 31.69  |
| 61440 Samples, stream | 258664.97  | 252.60  | 44  | 16.87  |
| 61440 Samples, batch | 592881.59  | 578.99  | 100  | 59.33  |
| 122880 Samples, serial | 1671307.20  | 1632.14  | 70  | 64.99  |
| 122880 Samples, stream | 680508.84  | 664.56  | 37  | 15.25  |
| 122880 Samples, batch | 1004880.23  | 981.33  | 100  | 59.00  |

-----

### **7\. Direct CPU vs. GPU Speedup**

Finally, To provide a clear, baseline performance comparison, the `test_channel_simulation` benchmark was run. This tool directly compares the execution time of the sequential, `float`-based CPU pipeline against the baseline synchronous GPU pipeline (`run_channel_pipeline_cuda`). The following tests were executed on a GH200 server using the ATS memory model. The results demonstrate the performance gains achieved by offloading the channel simulation to the GPU.

| Channel Type | MIMO Config | Signal Length | CPU Pipeline (µs) | GPU Pipeline (µs) | Overall Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
Short Channel   | 1x1             | 30720           | 654.12               | 45.60                | 14.34          x
Short Channel   | 2x2             | 30720           | 2145.07              | 57.48                | 37.32          x
Short Channel   | 4x4             | 30720           | 7611.24              | 84.20                | 90.40          x
Short Channel   | 8x8             | 30720           | 28190.40             | 148.07               | 190.39         x
Short Channel   | 1x1             | 61440           | 1305.85              | 54.16                | 24.11          x
Short Channel   | 2x2             | 61440           | 4290.31              | 80.34                | 53.40          x
Short Channel   | 4x4             | 61440           | 15264.63             | 136.62               | 111.73         x
Short Channel   | 8x8             | 61440           | 56561.05             | 255.82               | 221.10         x
Short Channel   | 1x1             | 122880          | 2617.06              | 79.33                | 32.99          x
Short Channel   | 2x2             | 122880          | 8585.16              | 129.69               | 66.20          x
Short Channel   | 4x4             | 122880          | 30543.07             | 233.36               | 130.88         x
Short Channel   | 8x8             | 122880          | 113185.64            | 468.10               | 241.80         x
Long Channel    | 1x1             | 30720           | 1068.25              | 44.37                | 24.08          x
Long Channel    | 2x2             | 30720           | 3795.86              | 58.86                | 64.49          x
Long Channel    | 4x4             | 30720           | 14209.40             | 89.08                | 159.51         x
Long Channel    | 8x8             | 30720           | 54579.90             | 164.72               | 331.36         x
Long Channel    | 1x1             | 61440           | 2139.73              | 56.02                | 38.20          x
Long Channel    | 2x2             | 61440           | 7608.75              | 83.18                | 91.47          x
Long Channel    | 4x4             | 61440           | 28463.66             | 145.80               | 195.23         x
Long Channel    | 8x8             | 61440           | 109785.85            | 289.96               | 378.62         x
Long Channel    | 1x1             | 122880          | 4267.25              | 80.65                | 52.91          x
Long Channel    | 2x2             | 122880          | 15209.56             | 135.12               | 112.56         x
Long Channel    | 4x4             | 122880          | 56951.08             | 249.47               | 228.29         x
Long Channel    | 8x8             | 122880          | 222368.44            | 524.97               | 423.59         x
