#if PARALLEL_STREAM
#include <cstdio>

__global__ void test_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from test kernel!\n");
    }
}

extern "C" void run_test_kernel() {
    test_kernel<<<1,1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
#endif
