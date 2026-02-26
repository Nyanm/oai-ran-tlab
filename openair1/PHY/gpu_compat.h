#pragma once

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_NVIDIA__)
  #define GPU_USE_HIP 1
  #include <hip/hip_runtime.h>
#else
  #define GPU_USE_HIP 0
  #include <cuda_runtime.h>
#endif

#if GPU_USE_HIP
  #define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
	      hipLaunchKernelGGL(kernel, grid, block, shmem, stream, __VA_ARGS__)
  #define GPU_DEVICE __device__
  #define GPU_FORCEINLINE __forceinline__
  #define GPU_RESTRICT __restrict__
  typedef hipStream_t gpuStream_t;
  static inline const char* gpuGetErrorString(hipError_t e){ return hipGetErrorString(e); }
#else
  #define GPU_LAUNCH_KERNEL(kernel, grid, block, shmem, stream, ...) \
	        kernel<<<grid, block, shmem, stream>>>(__VA_ARGS__)
  #define GPU_DEVICE __device__
  #define GPU_FORCEINLINE __forceinline__
  #define GPU_RESTRICT __restrict__
  typedef cudaStream_t gpuStream_t;
  static inline const char* gpuGetErrorString(cudaError_t e){ return cudaGetErrorString(e); }
#endif
