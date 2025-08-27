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

#define COARSE_FACTOR 1
#define BLOCK_WIDTH 1024
#define BLOCK_WIDTH_PER_DIM 32
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
    
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;

    extern __shared__ float inp_shared[];

    unsigned int max_per_row_offset = BLOCK_WIDTH*BLOCK_WIDTH;
    unsigned int sum_per_row_offset = BLOCK_WIDTH*BLOCK_WIDTH + BLOCK_WIDTH;
    unsigned int i = threadIdx.y*BLOCK_WIDTH + threadIdx.x;

    if (row < n && col < m) {
        inp_shared[i] = inp[row*m + col];
        inp_shared[max_per_row_offset + row] = -MAXFLOAT;
        inp_shared[sum_per_row_offset + row] = 0.0f;
    }

    __syncthreads();

    if (row < n && col < m) {
        atomicMaxF32(&inp_shared[max_per_row_offset + row], inp_shared[i]);
    }

    __syncthreads();

    if (row < n && col < m) {
        atomicAdd(&inp_shared[sum_per_row_offset + row], exp(inp_shared[i]-inp_shared[max_per_row_offset + row]));
    }

    __syncthreads();

    if (row < n && col < m) {
        out[row*m + col] = exp(inp_shared[i]-inp_shared[max_per_row_offset + row])/inp_shared[sum_per_row_offset + row];
    }
}

void softmax_cuda_launcher(float *inp, float *out, const unsigned long n, const unsigned long m) {
    dim3 bd(BLOCK_WIDTH_PER_DIM, BLOCK_WIDTH_PER_DIM, 1);
    dim3 gd((m+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, (n+BLOCK_WIDTH_PER_DIM-1)/BLOCK_WIDTH_PER_DIM, 1);

    softmax_cuda<<<gd, bd, (BLOCK_WIDTH*BLOCK_WIDTH + 2*BLOCK_WIDTH)*sizeof(float)>>>(inp, out, n, m);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }
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