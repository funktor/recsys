#include "matgraph.h"

using namespace std;

void matpaths_par(uint64_t *a, const uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n*n), 
            [&a, n, k](tbb::blocked_range<size_t> r) {
            for (auto p = r.begin(); p < r.end(); p++) {
                uint64_t i = p/n;
                uint64_t j = p % n;
                a[i*n+j] |= (a[i*n + k] & a[k*n + j]);
            }
        });
    }
}

void matpaths_simd(uint64_t *a, const uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        for (uint64_t i = 0; i < n; i++) {
            uint64x2_t c = vdupq_n_u64(a[i*n + k]);
            for (uint64_t j = 0; j < n; j += 2) {
                if (j+2 > n) {
                    for (uint64_t h = j; h < n; h++) a[i*n + h] |= (a[i*n + k] & a[k*n + h]);
                }
                else {
                    uint64x2_t x = vld1q_u64(&a[k*n + j]);
                    uint64x2_t y = vld1q_u64(&a[i*n + j]);
                    x = vandq_u64(x, c);
                    y = vorrq_u64(y, x);
                    vst1q_u64(&a[i*n + j], y);
                }
            }
        }
    }
}



u_int8_t matsearch_par(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst) {
    uint64_t *b = new uint64_t[n*n];
    for (uint64_t i = 0; i < n*n; i++) b[i] = a[i];

    for (uint64_t k = 0; k < n; k++) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n*n), 
            [&b, n, k](tbb::blocked_range<size_t> r) {
            for (auto p = r.begin(); p < r.end(); p++) {
                uint64_t i = p/n;
                uint64_t j = p % n;
                b[i*n+j] |= (b[i*n + k] & b[k*n + j]);
            }
        });
        if (b[src*n + dst]) return 1;
    }
    return 0;
}

u_int8_t matsearch_simd(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst) {
    uint64_t *b = new uint64_t[n*n];
    for (uint64_t i = 0; i < n*n; i++) b[i] = a[i];

    for (uint64_t k = 0; k < n; k++) {
        for (uint64_t i = 0; i < n; i++) {
            uint64x2_t c = vdupq_n_u64(b[i*n + k]);
            for (uint64_t j = 0; j < n; j += 2) {
                if (j+2 > n) {
                    for (uint64_t h = j; h < n; h++) b[i*n + h] |= (b[i*n + k] & b[k*n + h]);
                }
                else {
                    uint64x2_t x = vld1q_u64(&b[k*n + j]);
                    uint64x2_t y = vld1q_u64(&b[i*n + j]);
                    x = vandq_u64(x, c);
                    y = vorrq_u64(y, x);
                    vst1q_u64(&b[i*n + j], y);
                }
                if (b[src*n + dst]) return 1;
            }
        }
    }
    
    return 0;
}


uint64_t mat_numpaths_par(const uint64_t *a, const uint64_t n, const uint64_t src, const uint64_t dst) {
    uint64_t *b = new uint64_t[n*n];
    for (uint64_t i = 0; i < n*n; i++) b[i] = a[i];

    for (uint64_t k = 0; k < n; k++) {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, n*n), 
            [&b, n, k](tbb::blocked_range<size_t> r) {
            for (auto p = r.begin(); p < r.end(); p++) {
                uint64_t i = p/n;
                uint64_t j = p % n;
                b[i*n+j] += b[i*n + k] * b[k*n + j];
            }
        });
    }

    return b[src*n + dst];
}

f_csr_matrix dist_matmul_csr(const f_csr_matrix a, const f_csr_matrix b, const uint32_t n, const uint32_t m, const uint32_t p) {
    struct triplets{
        uint32_t i;
        uint32_t j;
        uint64_t v;
    };

    struct pairs{
        uint32_t j;
        uint64_t v;
    };

    f_csr_matrix bt;
    
    bt.nrows = p;
    bt.ncols = m;
    bt.data = new uint64_t[b.indices_size];
    bt.indices = new uint32_t[b.indices_size];
    bt.indptr = new uint32_t[p+1];
    bt.indptr[0] = 0;

    std::vector<triplets> res;

    for (uint32_t i = 0; i < b.nrows; i++) {
        uint32_t s = b.indptr[i];
        uint32_t e = b.indptr[i+1];

        for (uint32_t q = s; q < e; q++) {
            uint32_t j = b.indices[q];
            uint64_t u = b.data[q];
            res.push_back({j, i, u});
        }
    }

    std::sort(res.begin(), res.end(), [](triplets x, triplets y) {return (x.i == y.i)?(x.j < y.j):(x.i < y.i);});

    uint32_t k = 0;
    uint32_t gmax = res.back().i;

    for (auto g : res) {
        bt.indices[k] = g.j;
        bt.data[k] = g.v;
        k++;
        bt.indptr[g.i+1] = k;
    }

    for (uint32_t i = gmax+1; i < p+1; i++) bt.indptr[i] = k;

    res.clear();

    for (uint32_t i = 0; i < a.nrows; i++) {
        uint32_t s_a = a.indptr[i];
        uint32_t e_a = a.indptr[i+1];

        pairs *a_res = new pairs[e_a-s_a];
        uint32_t p = 0;
        for (uint32_t q = s_a; q < e_a; q++) {
            uint32_t k = a.indices[q];
            uint64_t u = a.data[q];
            a_res[p++] = {k, u};
        }

        for (uint32_t j = 0; j < bt.nrows; j++) {
            uint32_t s_b = bt.indptr[j];
            uint32_t e_b = bt.indptr[j+1];

            uint32_t k1 = 0;
            uint32_t k2 = s_b;
            uint64_t s = LLONG_MAX;

            while (k1 < e_a-s_a && k2 < e_b) {
                if (a_res[k1].j == bt.indices[k2]) {
                    s = std::min(s, a_res[k1].v + bt.data[k2]);
                    k1++;
                    k2++;
                }
                else if (a_res[k1].j < bt.indices[k2]) {
                    s = std::min(s, a_res[k1].v);
                    k1++;
                }
                else {
                    s = std::min(s, bt.data[k2]);
                    k2++;
                }
            }

            while (k1 < e_a-s_a) {
                s = std::min(s, a_res[k1].v);
                k1++;
            }

            while (k2 < e_b) {
                s = std::min(s, bt.data[k2]);
                k2++;
            }

            if (s != 0) res.push_back({i, j, s});
        }
    }

    delete[] bt.data;
    delete[] bt.indices;
    delete[] bt.indptr;

    f_csr_matrix c;

    c.nrows = n;
    c.ncols = p;
    c.indices_size = res.size();
    c.data = new uint64_t[c.indices_size];
    c.indices = new uint32_t[c.indices_size];
    c.indptr = new uint32_t[n+1];
    c.indptr[0] = 0;

    k = 0;
    gmax = res.back().i;

    for (auto g : res) {
        c.indices[k] = g.j;
        c.data[k] = g.v;
        k++;
        c.indptr[g.i+1] = k;
    }

    for (uint32_t i = gmax+1; i < n+1; i++) c.indptr[i] = k;
    return c;
}

f_csr_matrix matmul_csr(const f_csr_matrix a, const f_csr_matrix b, const uint32_t n, const uint32_t m, const uint32_t p) {
    struct triplets{
        uint32_t i;
        uint32_t j;
        uint64_t v;
    };

    struct pairs{
        uint32_t j;
        uint64_t v;
    };

    f_csr_matrix bt;
    
    bt.nrows = p;
    bt.ncols = m;
    bt.data = new uint64_t[b.indices_size];
    bt.indices = new uint32_t[b.indices_size];
    bt.indptr = new uint32_t[p+1];
    bt.indptr[0] = 0;

    std::vector<triplets> res;

    for (uint32_t i = 0; i < b.nrows; i++) {
        uint32_t s = b.indptr[i];
        uint32_t e = b.indptr[i+1];

        for (uint32_t q = s; q < e; q++) {
            uint32_t j = b.indices[q];
            uint64_t u = b.data[q];
            res.push_back({j, i, u});
        }
    }

    std::sort(res.begin(), res.end(), [](triplets x, triplets y) {return (x.i == y.i)?(x.j < y.j):(x.i < y.i);});

    uint32_t k = 0;
    uint32_t gmax = res.back().i;

    for (auto g : res) {
        bt.indices[k] = g.j;
        bt.data[k] = g.v;
        k++;
        bt.indptr[g.i+1] = k;
    }

    for (uint32_t i = gmax+1; i < p+1; i++) bt.indptr[i] = k;

    res.clear();

    for (uint32_t i = 0; i < a.nrows; i++) {
        uint32_t s_a = a.indptr[i];
        uint32_t e_a = a.indptr[i+1];

        pairs *a_res = new pairs[e_a-s_a];
        uint32_t p = 0;
        for (uint32_t q = s_a; q < e_a; q++) {
            uint32_t k = a.indices[q];
            uint64_t u = a.data[q];
            a_res[p++] = {k, u};
        }

        for (uint32_t j = 0; j < bt.nrows; j++) {
            uint32_t s_b = bt.indptr[j];
            uint32_t e_b = bt.indptr[j+1];

            uint32_t k1 = 0;
            uint32_t k2 = s_b;
            uint64_t s = 0;

            while (k1 < e_a-s_a && k2 < e_b) {
                if (a_res[k1].j == bt.indices[k2]) {
                    s += a_res[k1].v * bt.data[k2];
                    k1++;
                    k2++;
                }
                else if (a_res[k1].j < bt.indices[k2]) {
                    k1++;
                }
                else {
                    k2++;
                }
            }

            if (s != 0) res.push_back({i, j, s});
        }
    }

    delete[] bt.data;
    delete[] bt.indices;
    delete[] bt.indptr;

    f_csr_matrix c;

    c.nrows = n;
    c.ncols = p;
    c.indices_size = res.size();
    c.data = new uint64_t[c.indices_size];
    c.indices = new uint32_t[c.indices_size];
    c.indptr = new uint32_t[n+1];
    c.indptr[0] = 0;

    k = 0;
    gmax = res.back().i;

    for (auto g : res) {
        c.indices[k] = g.j;
        c.data[k] = g.v;
        k++;
        c.indptr[g.i+1] = k;
    }

    for (uint32_t i = gmax+1; i < n+1; i++) c.indptr[i] = k;
    return c;
}