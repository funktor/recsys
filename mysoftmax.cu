#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define COARSE_FACTOR 4
#define BLOCK_WIDTH 1024
#define MODIFIED_BLOCK_WIDTH COARSE_FACTOR*BLOCK_WIDTH

__device__ 
static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void softmax_cuda(float *inp, float *out, const unsigned long n, const unsigned long m) {
    extern __shared__ float inp_shared[];
    extern __shared__ float max_per_row[];
    extern __shared__ float sum_per_row[];

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        unsigned int r = i/m;
        if (index < n*m) inp_shared[i] = inp[index]; 
        else inp_shared[i] = 0.0f;

        max_per_row[r] = -MAXFLOAT;
        sum_per_row[r] = 0.0f;
    }

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int r = i/m;
        atomicMax(&max_per_row[r], inp_shared[i]);
    }

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int r = i/m;
        atomicAdd(&sum_per_row[r], exp(inp_shared[i]-max_per_row[r]));
    }

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        unsigned int r = i/m;
        out[index] = exp(inp_shared[i]-max_per_row[r])/sum_per_row[r];
    }
}

void softmax_cuda_launcher(float *inp, float *out, const unsigned long n, const unsigned long m) {
    unsigned int num_blocks = int(ceil(float(n*m)/MODIFIED_BLOCK_WIDTH));
    unsigned int num_rows_per_block = int(ceil(float(MODIFIED_BLOCK_WIDTH)/m));
    softmax_cuda<<<num_blocks, BLOCK_WIDTH, (MODIFIED_BLOCK_WIDTH + 2*num_rows_per_block)*sizeof(float)>>>(inp, out, n, m);
    cudaDeviceSynchronize();
}