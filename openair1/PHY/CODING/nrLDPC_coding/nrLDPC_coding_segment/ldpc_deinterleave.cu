// deinterleave_u16.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
static __device__ __forceinline__ int16_t lo16(uint32_t x) { return (int16_t)(x & 0xFFFFu); }
static __device__ __forceinline__ int16_t hi16(uint32_t x) { return (int16_t)(x >> 16); }


__global__ void deinterleave_i16_2(int16_t** __restrict__ e,
                                   const int16_t** __restrict__ f,
                                   int EQm1,
				   int EQm2,
				   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int EQm = (r<r_firstE2) ? EQm1 : EQm2;
    if (g >= EQm) return;

    // f[g*2 + 0..1] in one 32-bit load (requires only 4B alignment)
    const uint32_t v = *reinterpret_cast<const uint32_t*>(f[r] + 2*g);

    int16_t* e0 = e[r];
    int16_t* e1 = e0 + EQm;
    e0[g] = lo16(v);
    e1[g] = hi16(v);
}

__global__ void deinterleave_i16_4(int16_t** __restrict__ e,
                                   const int16_t** __restrict__ f,
                                   int EQm1,
                       		   int EQm2,
				   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int EQm = (r<r_firstE2) ? EQm1 : EQm2;
    if (g >= EQm) return;

    // 4x int16 = 8 bytes (requires 8B alignment for best perf; correctness works anyway)
    const uint2 v = *reinterpret_cast<const uint2*>(f[r] + 4*g);

    const uint32_t v0 = v.x; // lanes 0,1
    const uint32_t v1 = v.y; // lanes 2,3

    int16_t* e0 = e[r];
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;

    e0[g] = lo16(v0);
    e1[g] = hi16(v0);
    e2[g] = lo16(v1);
    e3[g] = hi16(v1);
}

__global__ void deinterleave_i16_6(int16_t** __restrict__ e,
                                   const int16_t** __restrict__ f,
                                   int EQm1,
								   int EQm2,
								   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
	int r = (int)blockIdx.y;
	int EQm = (r<r_firstE2) ? EQm1 : EQm2;
    if (g >= EQm) return;

    // 6x int16 = 12 bytes. Group stride is 12 => always 4B-aligned if f is 4B-aligned.
    const int16_t* in = f[r] + 6*g;
    const uint32_t a = *reinterpret_cast<const uint32_t*>(in + 0); // 0,1
    const uint32_t b = *reinterpret_cast<const uint32_t*>(in + 2); // 2,3
    const uint32_t c = *reinterpret_cast<const uint32_t*>(in + 4); // 4,5

    int16_t* e0 = e[r];
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;
    int16_t* e4 = e3 + EQm;
    int16_t* e5 = e4 + EQm;

    e0[g] = lo16(a);
    e1[g] = hi16(a);
    e2[g] = lo16(b);
    e3[g] = hi16(b);
    e4[g] = lo16(c);
    e5[g] = hi16(c);
}

__global__ void deinterleave_i16_8(int16_t** __restrict__ e,
                                   const int16_t** __restrict__ f,
                                   const int EQm1,
				   const int EQm2,
				   const int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int EQm = (r<r_firstE2) ? EQm1 : EQm2;
    if (g >= EQm) return;
	
    // 8x int16 = 16 bytes. Group stride is 16, so if f is 16B-aligned,
    // every group start is 16B-aligned and int4 loads are ideal.
    const int4 v = *reinterpret_cast<const int4*>(f[r] + 8*g);

    const uint32_t a = (uint32_t)v.x; // 0,1
    const uint32_t b = (uint32_t)v.y; // 2,3
    const uint32_t c = (uint32_t)v.z; // 4,5
    const uint32_t d = (uint32_t)v.w; // 6,7

    int16_t* e0 = e[r];
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;
    int16_t* e4 = e3 + EQm;
    int16_t* e5 = e4 + EQm;
    int16_t* e6 = e5 + EQm;
    int16_t* e7 = e6 + EQm;

    e0[g] = lo16(a);
    e1[g] = hi16(a);
    e2[g] = lo16(b);
    e3[g] = hi16(b);
    e4[g] = lo16(c);
    e5[g] = hi16(c);
    e6[g] = lo16(d);
    e7[g] = hi16(d);
}

// Host launcher
extern "C" void launch_deinterleave_i16(int Qm, int E1, int E2, int C, int r_firstE2,int16_t** e, const int16_t** f)
{
    const int EQm1=E1/Qm;
    const int EQm2=E2/Qm;
    const int threads = 256;
    dim3 blocks((EQm2  + threads - 1) / threads,C);

    switch (Qm) {
        case 2: deinterleave_i16_2<<<blocks, threads>>>(e, f, EQm1,EQm2,r_firstE2); break;
        case 4: deinterleave_i16_4<<<blocks, threads>>>(e, f, EQm1,EQm2,r_firstE2); break;
        case 6: deinterleave_i16_6<<<blocks, threads>>>(e, f, EQm1,EQm2,r_firstE2); break;
        case 8: deinterleave_i16_8<<<blocks, threads>>>(e, f, EQm1,EQm2,r_firstE2); break;
        default: /* unsupported */ break;
    }
    cudaDeviceSynchronize();
}
