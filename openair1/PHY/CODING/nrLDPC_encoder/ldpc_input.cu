#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
/*
__device__ const uint32_t masks[4] = {0x80,0x8000,0x800000,0x80000000};
__global__ void ldpc_input_worker(uint32_t **input,uint32_t *cc[4],int block_length,int nseg) {

  int block_off = blockIdx.x*blockDim.x<<2;	
  int i2 = threadIdx.x<<2;
  uint32_t *out=cc[blockIdx.y] + block_off + i2;
  int nseg0 = (blockIdx.y << 5);
  int nseg1;
  if ((nseg0 + 32) <= nseg) nseg1 = nseg0+32;
  else nseg1 = nseg0 + (nseg&31);
  int bit_offset = i2+block_off;
  int uint32_offset = bit_offset>>5;  
  uint32_t mask = masks[(bit_offset&31)>>3]; 
  uint32_t mask0 = mask>>(bit_offset&7);bit_offset++;
  uint32_t mask1 = mask>>(bit_offset&7);bit_offset++;
  uint32_t mask2 = mask>>(bit_offset&7);bit_offset++;
  uint32_t mask3 = mask>>(bit_offset&7);
  uint32_t tmp,jmod;
  uint32_t otmp0,otmp1,otmp2,otmp3;
  if (bit_offset  < block_length) {
      tmp=input[nseg0][uint32_offset];
      otmp0 = ((tmp&mask0) > 0); 
      otmp1 = ((tmp&mask1) > 0); 
      otmp2 = ((tmp&mask2) > 0); 
      otmp3 = ((tmp&mask3) > 0); 
      for (int j=nseg0+1;j<nseg1;j++) {
	 tmp=input[j][uint32_offset];
	 jmod = j&31;
         otmp0 |= (((tmp&mask0) > 0)<<jmod); 
         otmp1 |= (((tmp&mask1) > 0)<<jmod); 
         otmp2 |= (((tmp&mask2) > 0)<<jmod); 
         otmp3 |= (((tmp&mask3) > 0)<<jmod); 
      }	      
      out[0]=otmp0;
      out[1]=otmp1;
      out[2]=otmp2;
      out[3]=otmp3;
  } 
}
*/

#define ITERATIONS 1
#define NTHREADS 384 
__device__ uint32_t masks[32] = {
	0x80,0x40,0x20,0x10,0x8,0x4,0x2,0x1,
	0x8000,0x4000,0x2000,0x1000,0x800,0x400,0x200,0x100,
	0x800000,0x400000,0x200000,0x100000,0x80000,0x40000,0x20000,0x10000,
	0x80000000,0x40000000,0x20000000,0x10000000,0x8000000,0x4000000,0x2000000,0x1000000};
__global__ void ldpc_input_worker(uint32_t **input,uint32_t *cc[4],int block_length,int nseg) {

  int block_off = blockIdx.x*blockDim.x*ITERATIONS;	
  int i2 = threadIdx.x*ITERATIONS;
  uint32_t *out=cc[blockIdx.y] + block_off + i2;
  int nseg0 = (blockIdx.y << 5);
  int nseg1;
  if ((nseg0 + 32) <= nseg) nseg1 = nseg0+32;
  else nseg1 = nseg0 + (nseg&31);
  int bit_offset = i2+block_off;
  int uint32_offset = bit_offset>>5;  
#if ITERATIONS==1
  uint32_t mask0 = masks[bit_offset&31];
#else
  uint32_t *mask = &masks[bit_offset&31]; 
  uint32_t mask0 = mask[0];
#endif
#if ITERATIONS==4 || ITERATIONS==8
  uint32_t mask1 = mask[1];
  uint32_t mask2 = mask[2];
  uint32_t mask3 = mask[3];
#endif
#if ITERATIONS==8
  uint32_t mask4 = mask[4];
  uint32_t mask5 = mask[5];
  uint32_t mask6 = mask[6];
  uint32_t mask7 = mask[7];
#endif
  uint32_t tmp,jmod;
  uint32_t otmp0;
#if ITERATIONS==4 || ITERATIONS==8
  uint32_t otmp1,otmp2,otmp3;
#endif
#if ITERATIONS==8
  uint32_t otmp4,otmp5,otmp6,otmp7;
#endif
  if (bit_offset  < block_length) {
      tmp=input[nseg0][uint32_offset];
      otmp0 = ((tmp&mask0) > 0); 
#if ITERATIONS==4 || ITERATIONS==8
      otmp1 = ((tmp&mask1) > 0); 
      otmp2 = ((tmp&mask2) > 0); 
      otmp3 = ((tmp&mask3) > 0); 
#endif
#if ITERATIONS==8
      otmp4 = ((tmp&mask4) > 0); 
      otmp5 = ((tmp&mask5) > 0); 
      otmp6 = ((tmp&mask6) > 0); 
      otmp7 = ((tmp&mask7) > 0);
#endif 
      for (int j=nseg0+1;j<nseg1;j++) {
	 tmp=input[j][uint32_offset];
	 jmod = j&31;
         otmp0 |= (((tmp&mask0) > 0)<<jmod); 
#if ITERATIONS==4 || ITERATIONS==8
         otmp1 |= (((tmp&mask1) > 0)<<jmod); 
         otmp2 |= (((tmp&mask2) > 0)<<jmod); 
         otmp3 |= (((tmp&mask3) > 0)<<jmod);
#endif 
#if ITERATIONS==8
         otmp4 |= (((tmp&mask4) > 0)<<jmod); 
         otmp5 |= (((tmp&mask5) > 0)<<jmod); 
         otmp6 |= (((tmp&mask6) > 0)<<jmod); 
         otmp7 |= (((tmp&mask7) > 0)<<jmod); 
#endif
      }	      
      out[0]=otmp0;
#if ITERATIONS==4 || ITERATIONS==8
      out[1]=otmp1;
      out[2]=otmp2;
      out[3]=otmp3;
#endif
#if ITERATIONS==8
      out[4]=otmp4;
      out[5]=otmp5;
      out[6]=otmp6;
      out[7]=otmp7;
#endif
  } 
}
__global__ void circcopy_c_worker(uint32_t **cc,uint32_t **c) {

  int s = blockIdx.x;	
  int i1 = blockIdx.y;
  int i  = threadIdx.x;
  uint32_t tmp;
  if (i<384) {
    tmp = cc[s][(i1*384) + i];	  
    c[s][2*i1*384 + i] = tmp;
    c[s][(2*i1+1)*384 + i] = tmp; 
  }
}

extern "C" int ldpc_input(uint32_t **input,uint32_t *cc[4],int block_length,int nseg) { 

 int numb = block_length/(NTHREADS*ITERATIONS);
 if ((block_length%(NTHREADS*ITERATIONS)) > 0) numb++;
 int ns = nseg>>5;
 if ((nseg&31)>0) ns++;

 dim3 numblocks(numb,ns);
 ldpc_input_worker<<<numblocks,NTHREADS>>>(input,cc,block_length,nseg);
 cudaError_t err=cudaPeekAtLastError();
 if (err!=cudaSuccess) {
    printf("ldpc_input : cuda error: %s (input %p, cc %p, block_length %d, nseg %d, numb %d, ns %d)\n",cudaGetErrorString(err),input,cc,block_length,nseg,numb,ns);
    exit(-1);
 }
 cudaDeviceSynchronize();
 return(0);
}

extern "C" int circcopy_c(uint32_t **cc,uint32_t **c,int n_inputs) { 

 dim3 numblocks(n_inputs,22);
 circcopy_c_worker<<<numblocks,384>>>(cc,c);
 cudaError_t err=cudaPeekAtLastError();
 if (err!=cudaSuccess) {
    printf("circcopy_c : cuda error: %s (cc %p, c %p, n_inputs %d)\n",cudaGetErrorString(err),cc,c,n_inputs);
    exit(-1);
 }
 cudaDeviceSynchronize();
 return(0);
}
