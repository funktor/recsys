# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8

from libc.stdint cimport uint64_t, uint32_t, uint8_t
from cpython cimport array
import numpy as np
cimport numpy as np
from scipy.sparse import csr_matrix

cdef extern from "matgraph.h":
    struct f_csr_matrix:
        uint32_t nrows
        uint32_t ncols
        uint64_t *data
        uint32_t *indptr
        uint32_t *indices
        uint32_t indices_size

    void matpaths_par(uint64_t *a, uint64_t n)
    void matpaths_simd(uint64_t *a, uint64_t n)
    uint8_t matsearch_par(uint64_t *a, uint64_t n, uint64_t src, uint64_t dst)
    uint8_t matsearch_simd(uint64_t *a, uint64_t n, uint64_t src, uint64_t dst)
    uint64_t mat_numpaths_par(uint64_t *a, uint64_t n, uint64_t src, uint64_t dst)
    f_csr_matrix dist_matmul_csr(f_csr_matrix a, f_csr_matrix b, uint32_t n, uint32_t m, uint32_t p)
    f_csr_matrix matmul_csr(f_csr_matrix a, f_csr_matrix b, uint32_t n, uint32_t m, uint32_t p)

def py_matpaths_par(np.ndarray[np.uint64_t, ndim=2, mode='c'] a, uint64_t n):
    cdef uint64_t *a_arr = <uint64_t *> a.data
    matpaths_par(a_arr, n)

def py_matpaths_simd(np.ndarray[np.uint64_t, ndim=2, mode='c'] a, uint64_t n):
    cdef uint64_t *a_arr = <uint64_t *> a.data
    matpaths_simd(a_arr, n)

def py_matsearch_par(np.ndarray[np.uint64_t, ndim=2, mode='c'] a, uint64_t n, uint64_t src, uint64_t dst):
    cdef uint64_t *a_arr = <uint64_t *> a.data
    return matsearch_par(a_arr, n, src, dst)

def py_matsearch_simd(np.ndarray[np.uint64_t, ndim=2, mode='c'] a, uint64_t n, uint64_t src, uint64_t dst):
    cdef uint64_t *a_arr = <uint64_t *> a.data
    return matsearch_simd(a_arr, n, src, dst)

def py_mat_numpaths_par(np.ndarray[np.uint64_t, ndim=2, mode='c'] a, uint64_t n, uint64_t src, uint64_t dst):
    cdef uint64_t *a_arr = <uint64_t *> a.data
    return mat_numpaths_par(a_arr, n, src, dst)

def py_dist_matmul_csr(a, b, uint32_t n, uint32_t m, uint32_t p):
    cdef np.ndarray[np.uint64_t, ndim=1] a_data_view = a.data
    cdef np.ndarray[np.int32_t, ndim=1] a_indices_view = a.indices
    cdef np.ndarray[np.int32_t, ndim=1] a_indptr_view = a.indptr

    cdef np.ndarray[np.uint64_t, ndim=1] b_data_view = b.data
    cdef np.ndarray[np.int32_t, ndim=1] b_indices_view = b.indices
    cdef np.ndarray[np.int32_t, ndim=1] b_indptr_view = b.indptr

    cdef uint64_t *a_data = <uint64_t *> a_data_view.data
    cdef uint32_t *a_indices = <uint32_t *> a_indices_view.data
    cdef uint32_t *a_indptr = <uint32_t *> a_indptr_view.data

    cdef uint64_t *b_data = <uint64_t *> b_data_view.data
    cdef uint32_t *b_indices = <uint32_t *> b_indices_view.data
    cdef uint32_t *b_indptr = <uint32_t *> b_indptr_view.data

    cdef f_csr_matrix a_csr_matrix = f_csr_matrix(n, m, a_data, a_indptr, a_indices, len(a.indices))
    cdef f_csr_matrix b_csr_matrix = f_csr_matrix(m, p, b_data, b_indptr, b_indices, len(b.indices))

    out = dist_matmul_csr(a_csr_matrix, b_csr_matrix, n, m, p)
    cdef size = out.indices_size
    cdef nrow = out.nrows+1
    
    return csr_matrix((np.asarray(<uint64_t[:size]> out.data), np.asarray(<uint32_t[:size]> out.indices), np.asarray(<uint32_t[:nrow]> out.indptr)), shape=(n,p))


def py_matmul_csr(a, b, uint32_t n, uint32_t m, uint32_t p):
    cdef np.ndarray[np.uint64_t, ndim=1] a_data_view = a.data
    cdef np.ndarray[np.int32_t, ndim=1] a_indices_view = a.indices
    cdef np.ndarray[np.int32_t, ndim=1] a_indptr_view = a.indptr

    cdef np.ndarray[np.uint64_t, ndim=1] b_data_view = b.data
    cdef np.ndarray[np.int32_t, ndim=1] b_indices_view = b.indices
    cdef np.ndarray[np.int32_t, ndim=1] b_indptr_view = b.indptr

    cdef uint64_t *a_data = <uint64_t *> a_data_view.data
    cdef uint32_t *a_indices = <uint32_t *> a_indices_view.data
    cdef uint32_t *a_indptr = <uint32_t *> a_indptr_view.data

    cdef uint64_t *b_data = <uint64_t *> b_data_view.data
    cdef uint32_t *b_indices = <uint32_t *> b_indices_view.data
    cdef uint32_t *b_indptr = <uint32_t *> b_indptr_view.data

    cdef f_csr_matrix a_csr_matrix = f_csr_matrix(n, m, a_data, a_indptr, a_indices, len(a.indices))
    cdef f_csr_matrix b_csr_matrix = f_csr_matrix(m, p, b_data, b_indptr, b_indices, len(b.indices))

    out = matmul_csr(a_csr_matrix, b_csr_matrix, n, m, p)
    cdef size = out.indices_size
    cdef nrow = out.nrows+1
    
    return csr_matrix((np.asarray(<uint64_t[:size]> out.data), np.asarray(<uint32_t[:size]> out.indices), np.asarray(<uint32_t[:nrow]> out.indptr)), shape=(n,p))