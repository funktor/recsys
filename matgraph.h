#ifndef MATGRAPH_H
#define MATGRAPH_H

#include <tbb/tbb.h>
#include <arm_neon.h>
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

struct f_csr_matrix {
    uint32_t nrows;
    uint32_t ncols;
    uint64_t *data;
    uint32_t *indptr;
    uint32_t *indices;
    uint32_t indices_size;
};

void matpaths_par(uint64_t *a, const uint64_t n);
void matpaths_simd(uint64_t *a, const uint64_t n);
u_int8_t matsearch_par(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst);
u_int8_t matsearch_simd(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst);
uint64_t mat_numpaths_par(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst);
f_csr_matrix dist_matmul_csr(const f_csr_matrix a, const f_csr_matrix b, const uint32_t n, const uint32_t m, const uint32_t p);
f_csr_matrix matmul_csr(const f_csr_matrix a, const f_csr_matrix b, const uint32_t n, const uint32_t m, const uint32_t p);

#endif