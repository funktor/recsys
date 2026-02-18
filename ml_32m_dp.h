#ifndef ML_32M_DP_H
#define ML_32M_DP_H

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

struct user_ts {
    u_int32_t id;
    u_int32_t user_id;
    u_int64_t timestamps;
};

void get_historical_features(
                    const u_int32_t *user_id, 
                    const u_int32_t *movie_id, 
                    const u_int64_t *timestamps, 
                    const float32_t *normalized_ratings, 
                    u_int32_t **&prev_movie_ids, 
                    float32_t **&prev_ratings, 
                    u_int32_t *&num_tokens_prev,
                    const u_int32_t num_rows,
                    const u_int32_t max_hist
                );

#endif