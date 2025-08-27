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

#define COARSE_FACTOR 8
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
void softmax_cuda(float *inp, float *out, const unsigned long n, const unsigned long m) {
    // unsigned int row_index = blockIdx.x*blockDim.x + threadIdx.x;

    // if (row_index < n) {
    //     float max_val = -MAXFLOAT;
    //     float sum_val = 0.0f;

    //     for (unsigned int i = 0; i < m; i++) {
    //         max_val = (max_val < inp[row_index*m+i])?inp[row_index*m+i]:max_val;
    //     }

    //     for (unsigned int i = 0; i < m; i++) {
    //         sum_val += exp(inp[row_index*m+i]-max_val);
    //     }

    //     for (unsigned int i = 0; i < m; i++) {
    //         out[row_index*m+i] = exp(inp[row_index*m+i]-max_val)/sum_val;
    //     }
    // }
    

    extern __shared__ float inp_shared[];
    extern __shared__ float max_per_row_shared[];
    extern __shared__ float sum_per_row_shared[];

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        unsigned int r = index/m;

        if (index < n*m) {
            inp_shared[i] = inp[index]; 
            max_per_row_shared[r] = -MAXFLOAT;
            sum_per_row_shared[r] = 0.0f;
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            atomicMax(&max_per_row_shared[r], inp_shared[i]);
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            atomicAdd(&sum_per_row_shared[r], exp(inp_shared[i]-max_per_row_shared[r]));
        }
    }

    __syncthreads();

    for (unsigned int i = threadIdx.x; i < COARSE_FACTOR*blockDim.x; i += blockDim.x) {
        unsigned int index = COARSE_FACTOR*blockIdx.x*blockDim.x + i;
        if (index < n*m) {
            unsigned int r = index/m;
            out[index] = exp(inp_shared[i]-max_per_row_shared[r])/sum_per_row_shared[r];
        }
    }
}

void softmax_cuda_launcher(float *inp, float *out, const unsigned long n, const unsigned long m) {
    unsigned int num_blocks = int(ceil(float(n*m)/MODIFIED_BLOCK_WIDTH));
    unsigned int num_rows_per_block = int(ceil(float(MODIFIED_BLOCK_WIDTH)/m));
    softmax_cuda<<<num_blocks, BLOCK_WIDTH, (MODIFIED_BLOCK_WIDTH + 2*num_rows_per_block)*sizeof(float)>>>(inp, out, n, m);
    cudaDeviceSynchronize();

    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    // }
}

int main(int argc, char *argv[]) {
    unsigned int n = 10000;
    unsigned int m = 1000;

    float *x, *y;

    cudaMallocManaged(&x, n*m*sizeof(float));
    cudaMallocManaged(&y, n*m*sizeof(float));

    generate_data(x, n, m);

    auto start = std::chrono::high_resolution_clock::now();
    softmax_cuda_launcher(x, y, n, m);
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Duration = " << duration.count() << " ms" << std::endl;
}