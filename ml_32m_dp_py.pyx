# cython: language_level=3
# cython: c_string_type=unicode, c_string_encoding=utf8

from libc.stdint cimport uint64_t, uint32_t, uint8_t
from cpython cimport array
import numpy as np
cimport numpy as np
from libcpp cimport nullptr, nullptr_t
from libc.stdlib cimport malloc, free

cdef extern from "ml_32m_dp.h":
    void get_historical_features(
                    const uint32_t *user_id, 
                    const uint32_t *movie_id, 
                    const uint64_t *timestamps, 
                    const float *normalized_ratings, 
                    uint32_t **prev_movie_ids, 
                    float **prev_ratings, 
                    uint32_t *num_tokens_prev,
                    const uint32_t num_rows,
                    const uint32_t max_hist
                )

def py_get_historical_features(
                np.ndarray[np.uint32_t, ndim=1, mode='c'] user_id, 
                np.ndarray[np.uint32_t, ndim=1, mode='c'] movie_id,
                np.ndarray[np.uint64_t, ndim=1, mode='c'] timestamps, 
                np.ndarray[np.float32_t, ndim=1, mode='c'] normalized_ratings,
                uint32_t num_rows,
                uint32_t max_hist):

    cdef uint32_t **prev_movie_ids = NULL;
    cdef float **prev_ratings = NULL;
    cdef uint32_t *num_tokens_prev = NULL;

    cdef uint32_t *user_id_arr = <uint32_t *> user_id.data;
    cdef uint32_t *movie_id_arr = <uint32_t *> movie_id.data;
    cdef uint64_t *timestamps_arr = <uint64_t *> timestamps.data;
    cdef float *normalized_ratings_arr = <float *> normalized_ratings.data;

    get_historical_features(
        user_id_arr, 
        movie_id_arr, 
        timestamps_arr, 
        normalized_ratings_arr, 
        prev_movie_ids, 
        prev_ratings, 
        num_tokens_prev, 
        num_rows, 
        max_hist
    )

    py_prev_movie_ids = [[] for _ in range(num_rows)]
    py_prev_ratings   = [[] for _ in range(num_rows)]

    for i in range(num_rows):
        k = num_tokens_prev[i]
        for j in range(k):
            py_prev_movie_ids[i] += [prev_movie_ids[i][j]]
            py_prev_ratings[i]   += [prev_ratings[i][j]]

    return py_prev_movie_ids, py_prev_ratings
            
