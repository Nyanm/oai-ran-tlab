#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(int id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Hello from GPU kernel, id=%d\n", id);
    }
}

int main() {
    // 查询 GPU 属性
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Using device %d: %s (compute capability %d.%d)\n",
           device, prop.name, prop.major, prop.minor);

    // 启动 kernel
    hello_kernel<<<1,1>>>(42);

    // 检查 launch 是否成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 同步，等待 GPU 执行完
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Kernel finished successfully!\n");
    return 0;
}
