# GPU-Accelerated Channel Simulation

## 1\. Overview

This document describes the CUDA-based GPU acceleration pipeline for the OAI channel simulation. The primary goal of this feature is to offload the computationally intensive `multipath_channel` convolution and `add_noise` functions from the CPU to the GPU. This overcomes existing performance bottlenecks and enables large-scale, real-time physical layer simulation that is not feasible with the CPU-based models.

The result is a complete, high-performance pipeline integrated into the OAI simulators. The feature is supported by a new benchmark suite created to validate correctness and analyze performance across a wide range of channel models and hardware configurations.

## 2\. Implementation

This feature is implemented across a set of new CUDA source files and integrated into the existing OAI simulation framework.

### 2.1 Multipath Channel Convolution (`multipath_channel.cu`)

The core of the channel simulation is the multipath convolution. On the CPU, this is a sequential process where each output sample is calculated by iterating through all channel taps and accumulating the results. To parallelize this on the GPU, we assign one CUDA thread to calculate one output sample.

The key to making this parallel approach efficient is the use of a tiled convolution pattern with `__shared__` memory. Instead of having every thread read from slow global memory for each step of the convolution, threads in a block cooperate to pre-fetch all the data they will need into the on-chip shared memory.

The first step inside the kernel is for all threads in a block to collectively load a "tile" of the input signal. This load includes the "halo" of historical data needed from the previous block's logical space, ensuring the convolution is correct at the boundaries. The `__syncthreads()` call ensures no thread begins calculation until this shared data is fully loaded.

```cpp
// Snippet from multipath_channel_kernel in multipath_channel.cu
// Step 1: Threads in a block cooperate to load data into fast shared memory
for (int j = 0; j < nb_tx; j++) {
    const int tid = threadIdx.x;
    const int block_start_idx = blockIdx.x * blockDim.x;
    const int shared_mem_size = blockDim.x + channel_length - 1;

    // Each thread loads a piece of the signal from global into shared memory
    for (int k = tid; k < shared_mem_size; k += blockDim.x) {
        int load_idx = block_start_idx + k - (channel_length - 1);
        if (load_idx >= 0 && load_idx < num_samples) {
            tx_shared[k]= tx_sig[j * num_samples + load_idx];
        } else {
            // This zero-padding only happens at the very start of the signal (block 0)
            tx_shared[k]= make_float2(0.0f, 0.0f);
        }
    }
    // Wait for all threads to finish loading before proceeding
    __syncthreads();
```

Once the data is staged in fast shared memory, each thread can perform the full convolution for its assigned output sample. The inner loop iterates through the `channel_length` taps, but now all reads of the input signal (`tx_sample`) come from the fast `tx_shared` buffer. This significantly reduces memory latency and is the key to the kernel's performance. After the loop, the final accumulated result is written to global memory.

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

-----

## 3\. Memory Management Models

To provide flexibility and enable performance testing on different hardware architectures, the project now supports three distinct memory management models. The desired model is selected at build time by passing a flag to the CMake build command.

### 3.1 ATS Hybrid Model (`-DUSE_ATS_MEMORY=ON`)

This is a hybrid approach that leverages Address Translation Services (ATS), a hardware feature allowing the GPU to directly access host-allocated memory. In our implementation, the large input signal buffer is allocated in host memory using `malloc`. The GPU kernel is then able to read this data directly from the host, eliminating the need for an initial Host-to-Device `cudaMemcpy` for that buffer. Intermediate and output buffers are still allocated on the device, and the final result is copied back to the host explicitly. Due to its better performance compared to the other methods, it's also the default behaviour when no flag is specified.

### 3.2 Unified Memory Model (`-DUSE_UNIFIED_MEMORY=ON`)

This model simplifies memory management by creating a single, managed memory space that is accessible to both the CPU and the GPU. Memory is allocated once using `cudaMallocManaged`. The CUDA runtime and driver then automatically handle the migration of data to whichever processor is accessing it (e.g., moving data to the GPU's memory when a kernel is launched). This removes the need for most explicit `cudaMemcpy` calls. While the CUDA driver automatically migrates data on-demand, the implementation includes further optimizations to guide this process for better performance. `cudaMemAdvise` is used to provide performance hints to the driver, such as setting a buffer's preferred location to the GPU or marking it as mostly read-only. Additionally, `cudaMemPrefetchAsync` is also used to explicitly move data to the GPU's memory ahead of time. This ensures the data is already local when a kernel begins execution, which helps to hide the latency of the data migration.

### 3.3 Explicit Copy Model

This is the traditional CUDA programming model. The host and device have separate and distinct memory spaces. The programmer is responsible for all data management. Memory must be allocated on the device using `cudaMalloc`, and data must be manually transferred between the host and device using explicit `cudaMemcpy` calls before and after a kernel launch. This configuration can be run with other flags turned off by using `-DUSE_ATS_MEMORY=OFF` and `-DUSE_UNIFIED_MEMORY=OFF`.

-----

## 4\. Project Integration

To ensure the feature is modular, the integration into the OAI project was handled at both the build system and source code levels. All of the new CUDA source files (`channel_pipeline.cu`, `multipath_channel.cu`, `phase_noise.cu`) are compiled into a single static library named `oai_cuda_lib`. This is defined in the main `CMakeLists.txt` file and is controlled by the `CUDA_ENABLE` CMake option, which is disabled by default. When this option is activated (`-DCUDA_ENABLE=ON`), the build system compiles the CUDA library and also defines a global `ENABLE_CUDA` preprocessor macro that is visible to the rest of the project.

This `ENABLE_CUDA` macro is then used within the simulator files, such as `nr_dlsim.c` and `nr_ulsim.c`, to conditionally compile all CUDA-related code. This approach allows for the inclusion of the `oai_cuda.h` header, the addition of the `--cuda` runtime flag, and the pre-allocation of GPU memory at startup, all without affecting the standard CPU-only build.

Inside the main processing loop, an `if (use_cuda)` statement acts as the primary runtime switch. To enable a direct comparison, timing measurement calls have been added to both the GPU and CPU execution paths. The CPU path was also refined to use the `float` versions of the channel functions, ensuring a fair comparison against the single-precision GPU code. To execute the simulation on the GPU, the `--cuda` flag must be provided at runtime; otherwise, the program defaults to this refined CPU implementation.

Finally, the build system links the `nr_dlsim` and `nr_ulsim` executables against the `oai_cuda_lib` only when the feature is enabled, creating a clean separation between the two codebases.

-----

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

#### **Test Suite 1: Scalability vs. Channel Count**

This test measures performance by varying the number of channels for each of the three processing modes.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 8193.51  | 512.09  | 7  | 1.33  |
| 16 Channels, stream | 4753.84  | 297.11  | 8  | 1.76  |
| 16 Channels, batch | 979.77  | 61.24  | 8  | 1.47  |
| 256 Channels, serial | 131963.79  | 515.48  | 18  | 12.46  |
| 256 Channels, stream | 78985.00  | 308.54  | 26  | 16.29  |
| 256 Channels, batch | 15200.80  | 59.38  | 53  | 16.79  |
| 1024 Channels, serial | 534686.22  | 522.15  | 31  | 17.57  |
| 1024 Channels, stream | 305686.64  | 298.52  | 32  | 24.37  |
| 1024 Channels, batch | 60723.49  | 59.30  | 85  | 42.97  |

-----

#### **Test Suite 2: Performance vs. Channel Complexity**

This test measures how performance is affected by increasing the channel length (`-l`) for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 505383.50  | 493.54  | 83  | 15.41  |
| 16 Length, stream | 300009.57  | 292.98  | 26  | 20.42  |
| 16 Length, batch | 46055.11  | 44.98  | 78  | 35.00  |
| 64 Length, serial | 547509.06  | 534.68  | 72  | 22.62  |
| 64 Length, stream | 309591.32  | 302.34  | 45  | 33.60  |
| 64 Length, batch | 91425.47  | 89.28  | 90  | 53.08  |
| 128 Length, serial | 621081.51  | 606.52  | 84  | 30.28  |
| 128 Length, stream | 313960.27  | 306.60  | 66  | 50.73  |
| 128 Length, batch | 154107.94  | 150.50  | 91  | 65.14  |

-----

#### **Test Suite 3: Performance vs. MIMO Configuration**

This test evaluates performance by changing the MIMO (e.g., `-t 2 -r 2`) configuration for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 263152.23  | 256.98  | 64  | 16.27  |
| 2x2 MIMO, stream | 157129.21  | 153.45  | 28  | 20.42  |
| 2x2 MIMO, batch | 22005.70  | 21.49  | 76  | 26.11  |
| 4x4 MIMO, serial | 516357.97  | 504.26  | 76  | 18.22  |
| 4x4 MIMO, stream | 302816.95  | 295.72  | 32  | 24.28  |
| 4x4 MIMO, batch | 60730.69  | 59.31  | 85  | 45.48  |
| 8x8 MIMO, serial | 1068642.28  | 1043.60  | 42  | 22.43  |
| 8x8 MIMO, stream | 597318.97  | 583.32  | 43  | 35.38  |
| 8x8 MIMO, batch | 188886.44  | 184.46  | 94  | 67.59  |

-----

#### **Test Suite 4: Performance vs. Signal Samples**

This test assesses performance as the number of signal samples (`-s`) is varied for a fixed channel count of 1024.

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 149926.40  | 146.41  | 92  | 19.84  |
| 30720 Samples, stream | 83278.04  | 81.33  | 39  | 23.47  |
| 30720 Samples, batch | 15439.23  | 15.08  | 78  | 17.29  |
| 61440 Samples, serial | 274822.12  | 268.38  | 45  | 18.69  |
| 61440 Samples, stream | 158001.71  | 154.30  | 34  | 24.33  |
| 61440 Samples, batch | 30456.81  | 29.74  | 80  | 26.45  |
| 122880 Samples, serial | 521364.15  | 509.14  | 31  | 17.31  |
| 122880 Samples, stream | 303145.51  | 296.04  | 32  | 24.72  |
| 122880 Samples, batch | 60730.18  | 59.31  | 84  | 42.00  |

#### Analysis and Key Findings

The Explicit Copy model represents the traditional, highest-effort approach to CUDA programming, where the developer has full control over memory management. The results show this control translates directly into the highest achievable performance under the right conditions.

  * **Dominance of Batch Mode**: In every test suite, the `batch` execution mode is unequivocally the fastest and most efficient strategy. It excels by minimizing kernel launch overhead and maximizing the amount of parallel work sent to the GPU, allowing it to reach high levels of utilization. At 1024 channels, it is on average 7 times faster than `serial` mode.
  * **Excellent Scalability**: All modes demonstrate predictable, linear scaling. However, the `batch` mode's efficiency grows with the problem size. As seen in Test Suites 1, 3, and 4, its average GPU utilization increases dramatically as the number of channels, MIMO complexity, or sample count grows. This shows that batching is the ideal strategy for large-scale simulations.
  * **Predictable Performance**: The performance of all modes scales linearly with increases in workload, whether from more channels, higher complexity, or larger data sizes. This makes performance predictable and reliable.

-----

### 6.2 **ATS hybrid Memory Model**

#### **Test Suite 1: Scalability vs. Channel Count**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 4409.34  | 275.58  | 8  | 1.12  |
| 16 Channels, stream | 7235.36  | 452.21  | 7  | 2.24  |
| 16 Channels, batch | 17219.80  | 1076.24  | 5  | 1.70  |
| 256 Channels, serial | 66566.25  | 260.02  | 27  | 14.62  |
| 256 Channels, stream | 117478.40  | 458.90  | 25  | 11.40  |
| 256 Channels, batch | 301073.59  | 1176.07  | 15  | 4.86  |
| 1024 Channels, serial | 288641.36  | 281.88  | 29  | 23.11  |
| 1024 Channels, stream | 467186.87  | 456.24  | 27  | 14.72  |
| 1024 Channels, batch | 1168308.91  | 1140.93  | 50  | 5.79  |

-----

#### **Test Suite 2: Performance vs. Channel Complexity**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 274456.68  | 268.02  | 31  | 19.67  |
| 16 Length, stream | 462478.63  | 451.64  | 13  | 11.54  |
| 16 Length, batch | 1149634.51  | 1122.69  | 67  | 4.79  |
| 64 Length, serial | 314485.42  | 307.11  | 38  | 31.08  |
| 64 Length, stream | 466267.77  | 455.34  | 38  | 20.90  |
| 64 Length, batch | 1208185.82  | 1179.87  | 82  | 8.09  |
| 128 Length, serial | 388670.49  | 379.56  | 48  | 40.42  |
| 128 Length, stream | 473012.76  | 461.93  | 39  | 33.25  |
| 128 Length, batch | 1298377.52  | 1267.95  | 100  | 12.70  |

-----

#### **Test Suite 3: Performance vs. MIMO Configuration**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 137999.57  | 134.77  | 54  | 21.33  |
| 2x2 MIMO, stream | 223801.44  | 218.56  | 16  | 12.26  |
| 2x2 MIMO, batch | 559918.94  | 546.80  | 18  | 4.01  |
| 4x4 MIMO, serial | 291881.75  | 285.04  | 29  | 23.08  |
| 4x4 MIMO, stream | 474824.17  | 463.70  | 18  | 13.91  |
| 4x4 MIMO, batch | 1164531.97  | 1137.24  | 73  | 5.71  |
| 8x8 MIMO, serial | 701226.36  | 684.79  | 32  | 28.77  |
| 8x8 MIMO, stream | 906592.36  | 885.34  | 24  | 21.31  |
| 8x8 MIMO, batch | 2505682.00  | 2446.96  | 100  | 9.55  |

-----

#### **Test Suite 4: Performance vs. Signal Samples**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 83778.55  | 81.81  | 64  | 23.12  |
| 30720 Samples, stream | 107708.37  | 105.18  | 33  | 18.55  |
| 30720 Samples, batch | 275776.54  | 269.31  | 14  | 5.44  |
| 61440 Samples, serial | 146773.25  | 143.33  | 32  | 22.43  |
| 61440 Samples, stream | 230431.47  | 225.03  | 32  | 16.22  |
| 61440 Samples, batch | 578100.76  | 564.55  | 28  | 5.31  |
| 122880 Samples, serial | 287130.19  | 280.40  | 29  | 23.32  |
| 122880 Samples, stream | 475092.39  | 463.96  | 29  | 14.13  |
| 122880 Samples, batch | 1180025.98  | 1152.37  | 69  | 5.92  |

#### Analysis and Key Findings

The ATS model allows the GPU to access host memory directly, which can simplify some programming tasks. However, the performance results show this comes with a severe and counter-intuitive trade-off.

  * **Performance Inversion**: The most critical finding is the complete inversion of the expected performance hierarchy. In every single test, the `serial` mode is the fastest, while the `batch` mode is by far the slowest. At 1024 channels, `serial` execution is 4 times faster than `batch`.
  * **Prohibitive Overhead**: The `batch` mode's poor performance is directly explained by its extremely low GPU utilization, which never rises above 15% and is often in the single digits. This indicates that the overhead of the GPU needing to use ATS to resolve addresses for a large, scattered batch of data is so high that the GPU spends most of its time idle and starved of work.
  * **Serial Mode's Advantage**: The `serial` mode's simple, sequential memory access pattern is a much better fit for the ATS model. It avoids the high-overhead scenarios created by `batch` and `stream` modes, allowing it to achieve both the best performance and the most effective GPU utilization. Its efficiency also scales up appropriately as the complexity of the workload increases.

-----

### 6.3 **Unified Memory Model**

#### **Test Suite 1: Scalability vs. Channel Count**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Channels, serial | 2,547.89  | 159.24  | 8  | 1.00  |
| 16 Channels, stream | 8,467.46  | 529.22  | 11  | 2.78  |
| 16 Channels, batch | 17,933.37  | 1,120.84  | 33  | 11.90  |
| 256 Channels, serial | 36,489.72  | 142.54  | 45  | 19.44  |
| 256 Channels, stream | 122,215.14  | 477.40  | 25  | 14.22  |
| 256 Channels, batch | 265,435.95  | 1,036.86  | 72  | 45.27  |
| 1024 Channels, serial | 145,824.44  | 142.41  | 70  | 39.26  |
| 1024 Channels, stream | 485,937.84  | 474.55  | 40  | 19.13  |
| 1024 Channels, batch | 1,093,417.26  | 1,067.79  | 100  | 53.90  |

-----

#### **Test Suite 2: Performance vs. Channel Complexity**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 16 Length, serial | 126,377.72  | 123.42  | 62  | 33.88  |
| 16 Length, stream | 486,645.30  | 475.24  | 38  | 15.46  |
| 16 Length, batch | 1,025,308.49  | 1,001.28  | 100  | 52.43  |
| 64 Length, serial | 180,904.63  | 176.66  | 74  | 46.03  |
| 64 Length, stream | 489,448.63  | 477.98  | 51  | 23.99  |
| 64 Length, batch | 1,066,329.49  | 1,041.34  | 100  | 52.35  |
| 128 Length, serial | 253,898.46  | 247.95  | 78  | 54.77  |
| 128 Length, stream | 498,752.22  | 487.06  | 65  | 36.35  |
| 128 Length, batch | 1,026,795.55  | 1,002.73  | 100  | 52.19  |

-----

#### **Test Suite 3: Performance vs. MIMO Configuration**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 2x2 MIMO, serial | 72,976.37  | 71.27  | 47  | 24.44  |
| 2x2 MIMO, stream | 236,974.79  | 231.42  | 44  | 14.68  |
| 2x2 MIMO, batch | 554,929.41  | 541.92  | 100  | 52.07  |
| 4x4 MIMO, serial | 145,077.08  | 141.68  | 72  | 38.98  |
| 4x4 MIMO, stream | 493,066.06  | 481.51  | 44  | 18.59  |
| 4x4 MIMO, batch | 1,059,954.19  | 1,035.11  | 100  | 53.76  |
| 8x8 MIMO, serial | 355,888.73  | 347.55  | 73  | 49.76  |
| 8x8 MIMO, stream | 2,156,820.95  | 2,106.27  | 93  | 75.51  |
| 8x8 MIMO, batch | 2,162,668.02  | 2,111.98  | 100  | 53.24  |

-----

#### **Test Suite 4: Performance vs. Signal Samples**

| Configuration | Total GPU Time (µs) | Average GPU Time (µs) | Peak GPU Util (%) | Average GPU Util (%) |
| :--- | :--- | :--- | :--- | :--- |
| 30720 Samples, serial | 52,200.24  | 50.98  | 56  | 24.31  |
| 30720 Samples, stream | 119,896.90  | 117.09  | 49  | 19.93  |
| 30720 Samples, batch | 345,276.56  | 337.18  | 100  | 56.30  |
| 61440 Samples, serial | 82,986.18  | 81.04  | 65  | 34.26  |
| 61440 Samples, stream | 239,306.97  | 233.70  | 38  | 17.97  |
| 61440 Samples, batch | 623,935.89  | 609.31  | 100  | 56.73  |
| 122880 Samples, serial | 143,835.82  | 140.46  | 82  | 39.66  |
| 122880 Samples, stream | 486,935.57  | 475.52  | 44  | 18.17  |
| 122880 Samples, batch | 1,059,553.98  | 1,034.72  | 100  | 54.13  |

#### Analysis and Key Findings

The Unified Memory model aims to simplify development by abstracting away manual data transfers. However, this convenience comes at a steep performance cost, driven by the overhead of on-demand data migration (page faulting).

  * **Performance Inversion**: Just like the ATS model, the `serial` mode is the most performant strategy, while `batch` is the slowest. At 1024 channels, `serial` is 7.5 times faster than `batch`.
  * **High but Inefficient GPU Utilization**: This model's key characteristic is the combination of slow performance and high GPU utilization in `batch` mode. In Test Suite 1, `batch` mode reaches an average utilization of 53.90%, yet it is the slowest option. As a result, one can speculate that the GPU is not idle but is actively stalled, waiting for the driver and OS to handle page faults and move data from the host to the device. Perhaps the GPU is "busy" being memory-bound.

  * **Performance Cliff**: The 8x8 MIMO test in Suite 4 shows `stream` and `batch` modes hitting a performance wall, with execution times over six times slower than `serial`. The high utilization (75% for `stream`) combined with the poor performance might indicate a system bottlenecked by memory access rather than computation.


-----

According to the current results, the optimal configuration is highly dependent on the project's goals. For peak throughput and efficiency, `Explicit Copy` with `batch` mode is the winner. For other use cases where development simplicity is prioritized over raw speed, the `ATS` or `Unified Memory` models can be used, but only in conjunction with the `serial` execution mode.

-----

### **7\. Direct CPU vs. GPU Speedup**

Finally, To provide a clear, baseline performance comparison, the `test_channel_simulation` benchmark was run. This tool directly compares the execution time of the sequential, `float`-based CPU pipeline against the baseline synchronous GPU pipeline (`run_channel_pipeline_cuda`). The following tests were executed on a GH200 server using the ATS memory model. The results demonstrate the performance gains achieved by offloading the channel simulation to the GPU.

| Channel Type | MIMO Config | Signal Length | CPU Pipeline (µs) | GPU Pipeline (µs) | Overall Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Short Channel | 1x1 | 30720 | 659.50 | 39.90 | 16.53x |
| Short Channel | 2x2 | 30720 | 2,128.55 | 51.12 | 41.63x |
| Short Channel | 4x4 | 30720 | 7,479.31 | 82.20 | 90.99x |
| Short Channel | 8x8 | 30720 | 27,890.97 | 152.66 | 182.70x |
| Short Channel | 1x1 | 61440 | 1,323.72 | 47.92 | 27.62x |
| Short Channel | 2x2 | 61440 | 4,273.59 | 74.97 | 57.00x |
| Short Channel | 4x4 | 61440 | 15,009.81 | 131.20 | 114.41x |
| Short Channel | 8x8 | 61440 | 57,439.04 | 265.50 | 216.34x |
| Short Channel | 1x1 | 122880 | 2,652.54 | 71.21 | 37.25x |
| Short Channel | 2x2 | 122880 | 8,561.21 | 119.10 | 71.88x |
| Short Channel | 4x4 | 122880 | 30,428.09 | 235.52 | 129.19x |
| Short Channel | 8x8 | 122880 | 115,868.66 | 563.21 | 205.73x |
| Long Channel | 1x1 | 30720 | 1,064.13 | 37.46 | 28.40x |
| Long Channel | 2x2 | 30720 | 3,714.54 | 52.48 | 70.78x |
| Long Channel | 4x4 | 30720 | 13,805.75 | 86.26 | 160.06x |
| Long Channel | 8x8 | 30720 | 53,931.77 | 166.81 | 323.31x |
| Long Channel | 1x1 | 61440 | 2,122.26 | 48.74 | 43.55x |
| Long Channel | 2x2 | 61440 | 7,438.67 | 76.77 | 96.89x |
| Long Channel | 4x4 | 61440 | 27,696.60 | 138.21 | 200.40x |
| Long Channel | 8x8 | 61440 | 110,000.14 | 297.74 | 369.45x |
| Long Channel | 1x1 | 122880 | 4,253.51 | 72.68 | 58.52x |
| Long Channel | 2x2 | 122880 | 14,913.97 | 124.50 | 119.79x |
| Long Channel | 4x4 | 122880 | 56,194.75 | 252.61 | 222.46x |
| Long Channel | 8x8 | 122880 | 221,955.03 | 622.86 | 356.35x |
