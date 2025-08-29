#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <tuple>
#include <map>
#include <fcntl.h>
#include <functional>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <ctime> 
#include <stdbool.h>    // bool type
#include <fstream>
#include <cmath>
#include <variant>
#include <omp.h>
#include <math.h>
#include <assert.h>

#define BLOCK_WIDTH_PER_DIM 32

__device__ __forceinline__ float atomicMaxF32(float *address, float val) {
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__
void softmax_cuda(const float *inp, float *out, const unsigned long n, const unsigned long m) {
    unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long p = (m+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM;

    __shared__ float max_per_row[BLOCK_WIDTH_PER_DIM];
    __shared__ float sum_per_row[BLOCK_WIDTH_PER_DIM];

    if (row < n) {
        if (threadIdx.x == 0) {
            max_per_row[threadIdx.y] = -MAXFLOAT;
            sum_per_row[threadIdx.y] = 0.0f;
        }
        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                atomicMaxF32(&max_per_row[threadIdx.y], inp[row*m + j]);
            }
        }

        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                atomicAdd(&sum_per_row[threadIdx.y], exp(inp[row*m + j]-max_per_row[threadIdx.y]));
            }
        }

        __syncthreads();

        for (unsigned long j = threadIdx.x; j < p*BLOCK_WIDTH_PER_DIM; j += BLOCK_WIDTH_PER_DIM) {
            if (j < m) {
                out[row*m + j] = exp(inp[row*m + j]-max_per_row[threadIdx.y])/sum_per_row[threadIdx.y];
            }
        }
    }
}

__global__
void softmax_cuda_grad(const float *grad, const float *fwd, float *out, const unsigned long n, const unsigned long m) {
    unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < m) {
        float p = fwd[row*m + col];
        float s = 0.0f;

        for (unsigned long j = 0; j < m; j++) {
            if (j == col) s += grad[row*m + j]*p*(1.0-p);
            else s += -grad[row*m + j]*fwd[row*m + j]*p;
        }

        out[row*m + col] = s;
    }
}

__global__
void softmax_cuda_grad_shared_mem(const float *grad, const float *fwd, float *out, const unsigned long n, const unsigned long m) {
    unsigned long row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned long col = blockIdx.x*blockDim.x + threadIdx.x;

    __shared__ float grad_shared[BLOCK_WIDTH_PER_DIM*BLOCK_WIDTH_PER_DIM];
    __shared__ float fwd_shared[BLOCK_WIDTH_PER_DIM*BLOCK_WIDTH_PER_DIM];

    if (row < n && col < m) {
        float p = fwd[row*m + col];
        float s = 0.0f;

        for (int ph = 0; ph < (m+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM; ph++) {
            if (ph*BLOCK_WIDTH_PER_DIM + threadIdx.x < m) {
                grad_shared[threadIdx.y*BLOCK_WIDTH_PER_DIM + threadIdx.x] = grad[row*m + ph*BLOCK_WIDTH_PER_DIM + threadIdx.x];
                fwd_shared[threadIdx.y*BLOCK_WIDTH_PER_DIM + threadIdx.x] = fwd[row*m + ph*BLOCK_WIDTH_PER_DIM + threadIdx.x];
            }

            __syncthreads();


            for (unsigned long j = 0; j < BLOCK_WIDTH_PER_DIM; j++) {
                unsigned long k = ph*BLOCK_WIDTH_PER_DIM + j;
                if (k == col) s += grad_shared[threadIdx.y*BLOCK_WIDTH_PER_DIM + j]*p*(1.0 - p);
                else s += -grad_shared[threadIdx.y*BLOCK_WIDTH_PER_DIM + j]*p*fwd_shared[threadIdx.y*BLOCK_WIDTH_PER_DIM + j];
            }

            __syncthreads();
        }

        out[row*m + col] = s;
    }
}

void softmax_cuda_launcher(const float *inp, float *out, const unsigned long n, const unsigned long m) {
    dim3 bd(BLOCK_WIDTH_PER_DIM, BLOCK_WIDTH_PER_DIM, 1);
    dim3 gd(1, (n+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, 1);

    softmax_cuda<<<gd, bd>>>(inp, out, n, m);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

void softmax_cuda_grad_launcher(const float *grad, const float *fwd, float *out, const unsigned long n, const unsigned long m) {
    dim3 bd(BLOCK_WIDTH_PER_DIM, BLOCK_WIDTH_PER_DIM, 1);
    dim3 gd((m+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, (n+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, 1);

    softmax_cuda_grad<<<gd, bd>>>(grad, fwd, out, n, m);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}