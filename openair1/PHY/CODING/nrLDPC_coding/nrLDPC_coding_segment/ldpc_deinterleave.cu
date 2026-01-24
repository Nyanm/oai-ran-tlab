// deinterleave_u16.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
/*
static __device__ __forceinline__ int16_t lo16(uint32_t x) { return (int16_t)(x & 0xFFFFu); }
static __device__ __forceinline__ int16_t hi16(uint32_t x) { return (int16_t)(x >> 16); }
*/

__global__ void deinterleave_i16_2(int16_t* __restrict__ e,
                                   const int16_t* __restrict__ f,
                                   int E1,
				   int E2,
				   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int E = (r<r_firstE2) ? E1 : E2;
    int EQm = E/2;
    if (g >= EQm) return;

    int r_off = r<r_firstE2 ? r*E1 : ((r_firstE2*E1)+(r-r_firstE2)*E2);
    // f[g*2 + 0..1] in one 32-bit load (requires only 4B alignment)
    //const uint32_t v = *reinterpret_cast<const uint32_t*>(f +r_off + 2*g);
    const int16_t *in = f +r_off + 2*g;

    int16_t* e0 = e + r_off;
    int16_t* e1 = e0 + EQm;
    /*
    e0[g] = lo16(v);
    e1[g] = hi16(v);
    */
    e0[g] = in[0];
    e1[g] = in[1];
    //if (r==1 && g<8) printf("r %d, r_off %d, g %d, EQm %d, f[%d] %d f[%d] %d\n",r,r_off,g,EQm, 2*g, in[0], 1+(2*g), in[1]);
}

__global__ void deinterleave_i16_4(int16_t* __restrict__ e,
                                   const int16_t* __restrict__ f,
                                   int E1,
                       		   int E2,
				   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int E = (r<r_firstE2) ? E1 : E2;
    int EQm = E/4;
    if (g >= EQm) return;

    int r_off = r<r_firstE2 ? r*E1 : ((r_firstE2*E1)+(r-r_firstE2)*E2);
    // 4x int16 = 8 bytes (requires 8B alignment for best perf; correctness works anyway)
/*
    const uint2 v = {.x=0,.y=0}; //*reinterpret_cast<const uint2*>(f + r_off + 4*g);

    const uint32_t v0 = v.x; // lanes 0,1
    const uint32_t v1 = v.y; // lanes 2,3
*/
/*
    const uint32_t v0 = *reinterpret_cast<const uint32_t*>(f + r_off + 4*g);
    const uint32_t v1 = *reinterpret_cast<const uint32_t*>(f + r_off + 2 + 4*g);
    */
    const int16_t *in = f + r_off + 4*g;
    int16_t* e0 = e + r_off;
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;
/*
    e0[g] = lo16(v0);
    e1[g] = hi16(v0);
    e2[g] = lo16(v1);
    e3[g] = hi16(v1);
    */
    e0[g] = in[0];
    e1[g] = in[1];
    e2[g] = in[2];
    e3[g] = in[3];
}

__global__ void deinterleave_i16_6(int16_t* __restrict__ e,
                                   const int16_t* __restrict__ f,
                                   int E1,
				   int E2,
				   int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int E = (r<r_firstE2) ? E1 : E2;
    int EQm = E/6;
    if (g >= EQm) return;

    int r_off = r<r_firstE2 ? r*E1 : (r_firstE2*E1)+(r-r_firstE2)*E2;
    // 6x int16 = 12 bytes. Group stride is 12 => always 4B-aligned if f is 4B-aligned.
    //if ((r & 1) > 0) printf("g %d, r %d, r_off %d (%d), r_firstE2 %d, EQm1 %d EQm2 %d\n",g,r,r_off,r_off&7,r_firstE2,EQm1,EQm2);
    const int16_t* in = f + r_off + 6*g;
    /*
    const uint32_t a = 0;//*reinterpret_cast<const uint32_t*>(in + 0); // 0,1
    const uint32_t b = 0;//*reinterpret_cast<const uint32_t*>(in + 2); // 2,3
    const uint32_t c = 0;//*reinterpret_cast<const uint32_t*>(in + 4); // 4,5
*/
    int16_t* e0 = e + r_off;
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;
    int16_t* e4 = e3 + EQm;
    int16_t* e5 = e4 + EQm;
/*
    e0[g] = lo16(a);
    e1[g] = hi16(a);
    e2[g] = lo16(b);
    e3[g] = hi16(b);
    e4[g] = lo16(c);
    e5[g] = hi16(c);
    */

    e0[g] = in[0];
    e1[g] = in[1];
    e2[g] = in[2];
    e3[g] = in[3];
    e4[g] = in[4];
    e5[g] = in[5];
}

__global__ void deinterleave_i16_8(int16_t* __restrict__ e,
                                   const int16_t* __restrict__ f,
                                   const int E1,
				   const int E2,
				   const int r_firstE2)
{
    int g = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int r = (int)blockIdx.y;
    int E = (r<r_firstE2) ? E1 : E2;
    int EQm = E/8;
    if (g >= EQm) return;
    int r_off = r<r_firstE2 ? r*E1 : (r_firstE2*E1)+(r-r_firstE2)*E2;
    // 8x int16 = 16 bytes. Group stride is 16, so if f is 16B-aligned,
    // every group start is 16B-aligned and int4 loads are ideal.
    /*
    const int4 v = *reinterpret_cast<const int4*>(f + r_off + 8*g);

    const uint32_t a = (uint32_t)v.x; // 0,1
    const uint32_t b = (uint32_t)v.y; // 2,3
    const uint32_t c = (uint32_t)v.z; // 4,5
    const uint32_t d = (uint32_t)v.w; // 6,7
*/

    const int16_t *in = (f + r_off + 8*g);
    int16_t* e0 = e + r_off;
    int16_t* e1 = e0 + EQm;
    int16_t* e2 = e1 + EQm;
    int16_t* e3 = e2 + EQm;
    int16_t* e4 = e3 + EQm;
    int16_t* e5 = e4 + EQm;
    int16_t* e6 = e5 + EQm;
    int16_t* e7 = e6 + EQm;
/*
    e0[g] = lo16(a);
    e1[g] = hi16(a);
    e2[g] = lo16(b);
    e3[g] = hi16(b);
    e4[g] = lo16(c);
    e5[g] = hi16(c);
    e6[g] = lo16(d);
    e7[g] = hi16(d);*/
    e0[g] = in[0];
    e1[g] = in[1];
    e2[g] = in[2];
    e3[g] = in[3];
    e4[g] = in[4];
    e5[g] = in[5];
    e6[g] = in[6];
    e7[g] = in[7];
}

// Host launcher
extern "C" void launch_deinterleave_i16(int Qm, int E1, int E2, int C, int r_firstE2,int16_t* e, const int16_t* f,cudaStream_t *s,int8_t sidx)
{
    const int threads = 256;
    dim3 blocks(((E2/Qm)  + threads - 1) / threads,C);

    switch (Qm) {
        case 2: deinterleave_i16_2<<<blocks, threads, 0, s[sidx]>>>(e, f, E1,E2,r_firstE2); break;
        case 4: deinterleave_i16_4<<<blocks, threads, 0, s[sidx]>>>(e, f, E1,E2,r_firstE2); break;
        case 6: deinterleave_i16_6<<<blocks, threads, 0, s[sidx]>>>(e, f, E1,E2,r_firstE2); break;
        case 8: deinterleave_i16_8<<<blocks, threads, 0, s[sidx]>>>(e, f, E1,E2,r_firstE2); break;
        default: /* unsupported */ break;
    }
    cudaError_t err=cudaPeekAtLastError();
  
    if (err!=cudaSuccess) {
      printf("cuda error: %s (e %p, f %p, E1 %d, E2 %d, Qm %d, C %d)\n",cudaGetErrorString(err),e,f,E1,E2,Qm,C);
      exit(-1);
    }
    cudaDeviceSynchronize();
}
