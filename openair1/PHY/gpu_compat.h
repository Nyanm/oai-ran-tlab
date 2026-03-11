#pragma once

/*
  gpu_compat.h  --  C-compatible CUDA/HIP runtime abstraction

  Compile with:
    NVCC  -> CUDA
    HIPCC -> HIP

  No C++ features used.
*/

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
  #define GPU_USE_HIP 1
#else //if defined(__CUDACC__)
  #define GPU_USE_CUDA 1
//#else
//  #error "Compile with NVCC or HIPCC."
#endif
/* ================= LLR Type ================= */
#define INT16LLR 1  //0 to use int8_t llr, 1 to use int16_t llr

/* ================= Backend Includes ================= */

#if defined(GPU_USE_HIP)
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#elif defined(GPU_USE_CUDA)
  #include <cuda_runtime.h>
  #include <cuda_runtime_api.h>
#endif


/* ================= Common Qualifiers ================= */

#ifndef GPU_HOST
#define GPU_HOST __host__
#endif

#ifndef GPU_DEVICE
#define GPU_DEVICE __device__
#endif

#ifndef GPU_GLOBAL
#define GPU_GLOBAL __global__
#endif

#ifndef GPU_HOST_DEVICE
#define GPU_HOST_DEVICE __host__ __device__
#endif

#ifndef GPU_RESTRICT
#define GPU_RESTRICT __restrict__
#endif


/* ================= Type Abstraction ================= */

#if defined(GPU_USE_HIP)

typedef hipError_t    gpuError_t;
typedef hipStream_t   gpuStream_t;
typedef hipEvent_t    gpuEvent_t;

#define gpuSuccess hipSuccess

#define gpuMemcpyHostToHost     hipMemcpyHostToHost
#define gpuMemcpyHostToDevice   hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyDefault        hipMemcpyDefault

#elif defined(GPU_USE_CUDA)

typedef cudaError_t    gpuError_t;
typedef cudaStream_t   gpuStream_t;
typedef cudaEvent_t    gpuEvent_t;

#define gpuSuccess cudaSuccess

#define gpuMemcpyHostToHost     cudaMemcpyHostToHost
#define gpuMemcpyHostToDevice   cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost   cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyDefault        cudaMemcpyDefault

#endif


/* dim3 exists in both */
typedef dim3 gpuDim3;


/* ================= Error Handling ================= */

#include <stdio.h>
#include <stdlib.h>

#if defined(GPU_USE_HIP)
  #define gpuGetErrorName     hipGetErrorName
  #define gpuGetErrorString   hipGetErrorString
  #define gpuGetLastError     hipGetLastError
  #define gpuDeviceSynchronize hipDeviceSynchronize
  #define gpuPeekAtLastError  hipPeekAtLastError
#elif defined(GPU_USE_CUDA)
  #define gpuGetErrorName     cudaGetErrorName
  #define gpuGetErrorString   cudaGetErrorString
  #define gpuGetLastError     cudaGetLastError
  #define gpuDeviceSynchronize cudaDeviceSynchronize
  #define gpuPeekAtLastError  cudaPeekAtLastError
#endif

#ifndef GPU_CHECK
#define GPU_CHECK(call)                                                  \
  do {                                                                   \
    gpuError_t _e = (call);                                              \
    if (_e != gpuSuccess) {                                              \
      fprintf(stderr,                                                    \
        "GPU error: %s (%s) at %s:%d\n",                                 \
        gpuGetErrorName(_e), gpuGetErrorString(_e),                      \
        __FILE__, __LINE__);                                             \
      fflush(stderr);                                                    \
      abort();                                                           \
    }                                                                    \
  } while (0)
#endif

#define GPU_CHECK_LAUNCH() GPU_CHECK(gpuGetLastError())


/* ================= Runtime API Mapping ================= */

#if defined(GPU_USE_HIP)

#define gpuGetDeviceCount      hipGetDeviceCount
#define gpuGetDevice           hipGetDevice
#define gpuSetDevice           hipSetDevice
#define gpuGetDeviceProperties hipGetDeviceProperties

#define gpuMalloc              hipMalloc
#define gpuFree                hipFree
#define gpuMemset              hipMemset
#define gpuMemsetAsync         hipMemsetAsync
#define gpuHostAlloc           hipHostAlloc
#define gpuHostAllocMapped     hipHostAllocMapped
#define gpuHostAllocDefault    hipHostAllocDefault
#define gpuHostAllocPortable   hipHostAllocPortable
#define gpuFreeHost            hipFreeHost
#define gpuHostGetDevicePointer hipHostGetDevicePointer

#define gpuMemcpy              hipMemcpy
#define gpuMemcpyAsync         hipMemcpyAsync

#define gpuDeviceProp_t        hipDeviceProp_t
#define gpuDeviceGetAttribute  hipDeviceGetAttribute
#define gpuDevAttrManagedMemory hipDeviceAttributeManagedMemory
#define gpuDevAttrConcurrentManagedAccess hipDeviceAttributeConcurrentManagedAccess
#define gpuDevAttrUnifiedAddressing hipDeviceAttributeUnifiedAddressing
#define gpuDevAttrPageableMemoryAccess hipDeviceAttributePageableMemoryAccess
#define gpuDevAttrPageableMemoryAccessUsesHostPageTables hipDeviceAttributePageableMemoryAccessUsesHostPageTables
#define gpuDevAttrHostRegisterSupported hipDeviceAttributeHostRegisterSupported
#define gpuDevAttrIntegrated hipDeviceAttributeIntegrated


#define gpuStreamCreate        hipStreamCreate
#define gpuStreamDestroy       hipStreamDestroy
#define gpuStreamSynchronize   hipStreamSynchronize
#define gpuStreamNonBlocking   hipStreamNonBlocking
#define gpuStreamCreateWithFlags        hipStreamCreateWithFlags
#define gpuStreamBeginCapture  hipStreamBeginCapture
#define gpuStreamEndCapture    hipStreamEndCapture
#define gpuStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal

#define gpuEventCreate         hipEventCreate
#define gpuEventDestroy        hipEventDestroy
#define gpuEventRecord         hipEventRecord
#define gpuEventSynchronize    hipEventSynchronize
#define gpuEventElapsedTime    hipEventElapsedTime

#define gpuGraph_t             hipGraph_t
#define gpuGraphExec_t         hipGraphExec_t
#define gpuGraphInstantiate    hipGraphInstantiate
#define gpuGraphLaunch         hipGraphLaunch
#define gpuGraphDestroy        hipGraphDestroy

#elif defined(GPU_USE_CUDA)

#define gpuGetDeviceCount      cudaGetDeviceCount
#define gpuGetDevice           cudaGetDevice
#define gpuSetDevice           cudaSetDevice
#define gpuGetDeviceProperties cudaGetDeviceProperties

#define gpuDeviceProp_t        struct cudaDeviceProp
#define gpuDeviceGetAttribute  cudaDeviceGetAttribute
#define gpuDevAttrManagedMemory cudaDevAttrManagedMemory
#define gpuDevAttrConcurrentManagedAccess cudaDevAttrConcurrentManagedAccess
#define gpuDevAttrUnifiedAddressing cudaDevAttrUnifiedAddressing
#define gpuDevAttrPageableMemoryAccess cudaDevAttrPageableMemoryAccess
#define gpuDevAttrPageableMemoryAccessUsesHostPageTables cudaDevAttrPageableMemoryAccessUsesHostPageTables
#define gpuDevAttrHostRegisterSupported cudaDevAttrHostRegisterSupported
#define gpuDevAttrIntegrated   cudaDevAttrIntegrated

#define gpuMalloc              cudaMalloc
#define gpuFree                cudaFree
#define gpuMemset              cudaMemset
#define gpuMemsetAsync         cudaMemsetAsync
#define gpuHostAlloc           cudaHostAlloc
#define gpuHostAllocMapped     cudaHostAllocMapped
#define gpuHostAllocDefault    cudaHostAllocDefault
#define gpuHostAllocPortable   cudaHostAllocPortable
#define gpuFreeHost            cudaFreeHost
#define gpuHostGetDevicePointer cudaHostGetDevicePointer

#define gpuMemcpy              cudaMemcpy
#define gpuMemcpyAsync         cudaMemcpyAsync

#define gpuStreamCreate        cudaStreamCreate
#define gpuStreamCreateWithFlags        cudaStreamCreateWithFlags
#define gpuStreamDestroy       cudaStreamDestroy
#define gpuStreamSynchronize   cudaStreamSynchronize
#define gpuStreamNonBlocking   cudaStreamNonBlocking
#define gpuStreamBeginCapture  cudaStreamBeginCapture
#define gpuStreamEndCapture    cudaStreamEndCapture
#define gpuStreamCaptureModeThreadLocal cudaStreamCaptureModeThreadLocal

#define gpuEventCreate         cudaEventCreate
#define gpuEventDestroy        cudaEventDestroy
#define gpuEventRecord         cudaEventRecord
#define gpuEventSynchronize    cudaEventSynchronize
#define gpuEventElapsedTime    cudaEventElapsedTime

#define gpuGraph_t             cudaGraph_t
#define gpuGraphExec_t         cudaGraphExec_t
#define gpuGraphInstantiate    cudaGraphInstantiate
#define gpuGraphLaunch         cudaGraphLaunch
#define gpuGraphDestroy        cudaGraphDestroy
#endif


/* ================= Kernel Launch Abstraction ================= */

#if defined(GPU_USE_HIP)

  #define gpuLaunchKernel(kernel, grid, block, sharedBytes, stream, ...) \
      hipLaunchKernelGGL(kernel, grid, block, sharedBytes, stream, __VA_ARGS__)

#elif defined(GPU_USE_CUDA)

  #define gpuLaunchKernel(kernel, grid, block, sharedBytes, stream, ...) \
      do {                                                                \
        kernel<<<grid, block, sharedBytes, stream>>>(__VA_ARGS__);       \
      } while (0)

#endif
