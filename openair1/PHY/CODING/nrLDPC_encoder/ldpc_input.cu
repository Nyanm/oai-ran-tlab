#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

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
  if (bit_offset  < block_length) {
      for (int j=nseg0;j<nseg1;j++) {
	 tmp=input[j][uint32_offset];
	 jmod = j&31;
         *out     |= (((tmp&mask0) > 0)<<jmod); 
         *(out+1) |= (((tmp&mask1) > 0)<<jmod); 
         *(out+2) |= (((tmp&mask2) > 0)<<jmod); 
         *(out+3) |= (((tmp&mask3) > 0)<<jmod); 
      }	      
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
#define NTHREADS 768 
extern "C" int ldpc_input(uint32_t **input,uint32_t *cc[4],int block_length,int nseg) { 

 int numb = block_length/(NTHREADS*4);
 if ((block_length%(NTHREADS*4)) > 0) numb++;
 int ns = nseg>>5;
 if ((nseg&31)>0) ns++;

 dim3 numblocks(numb,ns);
 ldpc_input_worker<<<numblocks,NTHREADS>>>(input,cc,block_length,nseg);
 cudaError_t err=cudaPeekAtLastError();
 if (err!=cudaSuccess) {
    printf("cuda error: %s (input %p, cc %p, block_length %d, nseg %d, numb %d, ns %d)\n",cudaGetErrorString(err),input,cc,block_length,nseg,numb,ns);
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
    printf("cuda error: %s (cc %p, c %p, n_inputs %d)\n",cudaGetErrorString(err),cc,c,n_inputs);
    exit(-1);
 }
 cudaDeviceSynchronize();
 return(0);
}
