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

void softmax_cuda_launcher(const float *inp, float *out, const unsigned long n, const unsigned long m);
void softmax_cuda_grad_launcher(const float *grad, const float *fwd, float *out, const unsigned long n, const unsigned long m);

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
                    max_per_row[i] = std::max(max_per_row[i], inp[i*m+j]);
                }
            }
        });

        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&sum_per_row, &max_per_row, &inp, m](tbb::blocked_range<size_t> r) {
            for (auto i = r.begin(); i < r.end(); i++) {
                for (unsigned long j = 0; j < m; j++) {
                    sum_per_row[i] += exp(inp[i*m+j]-max_per_row[i]);
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

    void softmax_grad(const float *grad, const float *fwd, float *out, const unsigned long n, const unsigned long m) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n), 
            [&fwd, &out, &grad, m](tbb::blocked_range<size_t> r) {
                for (auto i = r.begin(); i < r.end(); i++) {
                    for (unsigned int j = 0; j < m; j++) {
                        float s = 0.0;
                        for (unsigned int k = 0; k < m; k++) {
                            if (j == k) s += grad[i*m + k]*fwd[i*m + j]*(1.0 - fwd[i*m + j]);
                            else s += -grad[i*m + k]*fwd[i*m + k]*fwd[i*m + j];
                        }
                        out[i*m + j] = s;
                    }
                }
            }
        );
    }

    torch::Tensor softmax_cpu(const torch::Tensor &a) {
        TORCH_CHECK(a.device().is_cpu(), "Input tensor a must be a CPU tensor");
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0);
        unsigned long m = a.size(1);
    
        softmax(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }


    torch::Tensor softmax_cpu_grad(const torch::Tensor &grad, const torch::Tensor &fwd_out) {
        TORCH_CHECK(fwd_out.device().is_cpu(), "Input tensor fwd_out must be a CPU tensor");
        TORCH_CHECK(grad.device().is_cpu(), "Input tensor grad must be a CPU tensor");

        TORCH_CHECK(fwd_out.is_contiguous(), "Input tensor fwd_out must be contiguous");
        TORCH_CHECK(grad.is_contiguous(), "Input tensor grad must be contiguous");

        TORCH_CHECK(fwd_out.dtype() == torch::kFloat32, "Input tensor fwd_out must be float32");
        TORCH_CHECK(grad.dtype() == torch::kFloat32, "Input tensor grad must be float32");

        TORCH_CHECK(grad.size(0) == fwd_out.size(0) && grad.size(1) == fwd_out.size(1), "Mismatched shapes");
    
        torch::Tensor c = torch::empty_like(grad);
        unsigned long n = grad.size(0);
        unsigned long m = grad.size(1);
    
        softmax_grad(
            grad.data_ptr<float>(),
            fwd_out.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

    torch::Tensor softmax_gpu(const torch::Tensor &a) {
        TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be a CUDA tensor");
        TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
        TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    
        torch::Tensor c = torch::empty_like(a);
        unsigned long n = a.size(0);
        unsigned long m = a.size(1);
    
        softmax_cuda_launcher(
            a.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

    torch::Tensor softmax_gpu_grad(const torch::Tensor &grad, const torch::Tensor &fwd_out) {
        TORCH_CHECK(fwd_out.device().is_cuda(), "Input tensor fwd_out must be a CUDA tensor");
        TORCH_CHECK(grad.device().is_cuda(), "Input tensor grad must be a CUDA tensor");

        TORCH_CHECK(fwd_out.is_contiguous(), "Input tensor fwd_out must be contiguous");
        TORCH_CHECK(grad.is_contiguous(), "Input tensor grad must be contiguous");

        TORCH_CHECK(fwd_out.dtype() == torch::kFloat32, "Input tensor fwd_out must be float32");
        TORCH_CHECK(grad.dtype() == torch::kFloat32, "Input tensor grad must be float32");

        TORCH_CHECK(grad.size(0) == fwd_out.size(0) && grad.size(1) == fwd_out.size(1), "Mismatched shapes");
    
        torch::Tensor c = torch::empty_like(grad);
        unsigned long n = grad.size(0);
        unsigned long m = grad.size(1);
    
        softmax_cuda_grad_launcher(
            grad.data_ptr<float>(),
            fwd_out.data_ptr<float>(),
            c.data_ptr<float>(),
            n, 
            m
        );
    
        return c;
    }

    PYBIND11_MODULE(extension_cpp, m) {
        m.def("mysoftmax_cpu", &softmax_cpu, "Softmax CPU Forward");
        m.def("mysoftmax_gpu", &softmax_gpu, "Softmax GPU Forward");
        m.def("mysoftmax_cpu_grad", &softmax_cpu_grad, "Softmax CPU Backward");
        m.def("mysoftmax_gpu_grad", &softmax_gpu_grad, "Softmax GPU Backward");
    }
}
