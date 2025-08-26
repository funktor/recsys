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

#define COARSE_FACTOR 4
#define BLOCK_WIDTH 1024
#define MODIFIED_BLOCK_WIDTH COARSE_FACTOR*BLOCK_WIDTH

void generate_data(float *x, unsigned int n, unsigned int m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<float> dist(0.0, 1.0);

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < m; j++) x[i*m+j] = dist(engine);
    }
}

void print_vector(float *x, int start, int end) {
    std::cout << "[";
    for (int i = start; i <= end; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

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
void softmax_cuda(float *inp, float *max_per_row, float *sum_per_row, float *out, const unsigned long n, const unsigned long m) {
    extern __shared__ float inp_shared[];

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        unsigned int r = index/m;

        if (index < n*m) {
            inp_shared[i] = inp[index]; 
            max_per_row[r] = -MAXFLOAT;
            sum_per_row[r] = 0.0f;
        }
        else inp_shared[i] = 0.0f;
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            atomicMax(&max_per_row[r], inp_shared[i]);
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            atomicAdd(&sum_per_row[r], exp(inp_shared[i]-max_per_row[r]));
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            out[index] = exp(inp_shared[i]-max_per_row[r])/sum_per_row[r];
        }
    }
}

void softmax_cuda_launcher(float *inp, float *out, const unsigned long n, const unsigned long m) {
    float *max_per_row, *sum_per_row;

    cudaMallocManaged(&max_per_row, n*sizeof(float));
    cudaMallocManaged(&sum_per_row, n*sizeof(float));

    unsigned int num_blocks = int(ceil(float(n*m)/MODIFIED_BLOCK_WIDTH));
    softmax_cuda<<<num_blocks, BLOCK_WIDTH, MODIFIED_BLOCK_WIDTH*sizeof(float)>>>(inp, max_per_row, sum_per_row, out, n, m);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
}

int main(int argc, char *argv[]) {
    unsigned int n = 1000;
    unsigned int m = 100;

    float *x, *y;

    cudaMallocManaged(&x, n*m*sizeof(float));
    cudaMallocManaged(&y, n*m*sizeof(float));

    generate_data(x, n, m);
    print_vector(x, 0, 100);

    softmax_cuda_launcher(x, y, n, m);
    print_vector(y, 0, 100);
}
