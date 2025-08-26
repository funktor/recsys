#include <Python.h>
#include <torch/extension.h>
#include <tbb/tbb.h>
#include <unistd.h>
#include <stdio.h>
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
#include <assert.h>
#include <initializer_list>
// #include "/opt/homebrew/opt/libomp/include/omp.h"

// extern "C" {
//     /* Creates a dummy empty _C module that can be imported from Python.
//        The import from Python will load the .so consisting of this file
//        in this extension, so that the TORCH_LIBRARY static initializers
//        below are run. */
//     PyObject* PyInit__C(void)
//     {
//         static struct PyModuleDef module_def = {
//             PyModuleDef_HEAD_INIT,
//             "_C",   /* name of module */
//             NULL,   /* module documentation, may be NULL */
//             -1,     /* size of per-interpreter state of the module,
//                        or -1 if the module keeps state in global variables. */
//             NULL,   /* methods */
//         };
//         return PyModule_Create(&module_def);
//     }
// }

// void generate_data(float *x, unsigned int n, unsigned int m) {
//     std::random_device rd;
//     std::mt19937 engine(rd());

//     std::uniform_real_distribution<float> dist(0.0, 1.0);

//     for (unsigned int i = 0; i < n; i++) {
//         for (unsigned int j = 0; j < m; j++) x[i*m+j] = dist(engine);
//     }
// }

void softmax_cuda_launcher(float *inp, float *out, const unsigned long n, const unsigned long m);

namespace extension_cpp {
    void softmax(const float *inp, float *out, const unsigned long n, const unsigned long m) {
        float *max_per_row = new float[n];
        float *sum_per_row = new float[n];

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&max_per_row, &sum_per_row](tbb::blocked_range<size_t> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                max_per_row[i] = -MAXFLOAT;
                sum_per_row[i] = 0.0;
            }
        });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                for (unsigned long j = 0; j < m; j++) {
                    max_per_row[i] = std::max(max_per_row[i], inp[i]);
                }
            }
        });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&sum_per_row, &max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                for (unsigned long j = 0; j < m; j++) {
                    sum_per_row[i] += exp(inp[i]-max_per_row[i]);
                }
            }
        });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&sum_per_row, &max_per_row, &inp, &out, m](tbb::blocked_range<size_t> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                for (unsigned long j = 0; j < m; j++) {
                    out[i*m+j] = exp(inp[i*m+j]-max_per_row[i])/sum_per_row[i];
                }
            }
        });
    }

    torch::Tensor softmax_cpu(const torch::Tensor &a) {
        // Input validation
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    
        // Create the output tensor on the same device as input
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0); // Total number of elements
        unsigned long m = a.size(1);
    
        // Call the CUDA launcher function
        softmax(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

    torch::Tensor softmax_gpu(const torch::Tensor &a) {
        // Input validation
        TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    
        // Create the output tensor on the same device as input
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0); // Total number of elements
        unsigned long m = a.size(1);
    
        // Call the CUDA launcher function
        softmax_cuda_launcher(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

    PYBIND11_MODULE(extension_cpp, m) {
        m.def("mysoftmax", &softmax_cpu, "LLTM forward 4");
    }

    PYBIND11_MODULE(extension_cpp, m) {
        m.def("mysoftmax_gpu", &softmax_gpu, "LLTM forward 4");
    }

    // TORCH_LIBRARY(extension_cpp, m) {
    //     m.def("mysoftmax(Tensor a) -> Tensor");
    // }
    
    // TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
    //     m.impl("mysoftmax", &softmax_cpu);
    // }
}

// int main(int argc, char *argv[]) {
//     unsigned int n = 10000;
//     unsigned int m = 1000;

//     float *x = new float[n*m];
//     float *y = new float[n*m];
//     generate_data(x, n, m);

//     auto start = std::chrono::high_resolution_clock::now();
//     extension_cpp::softmax(x, y, n, m);
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

//     std::cout << "Duration = " << duration.count() << " ms" << std::endl;
// }

