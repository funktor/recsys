#include "ml_32m_dp.h"

using namespace std;

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
                ) {
        
    user_ts *uts = new user_ts[num_rows];

    for (unsigned int i = 0; i < num_rows; i++) uts[i] = {i, user_id[i], timestamps[i]};
    std::sort(uts, uts + num_rows, [](const user_ts &a, const user_ts &b){return (a.user_id == b.user_id)?(a.timestamps < b.timestamps):(a.user_id < b.user_id);});

    prev_movie_ids = new u_int32_t*[num_rows];
    prev_ratings   = new float32_t*[num_rows];

    num_tokens_prev = new u_int32_t[num_rows];
    for (unsigned int i = 0; i < num_rows; i++) num_tokens_prev[i] = 0;

    std::vector<u_int32_t> p_m_ids;
    std::vector<float32_t> p_r_ids;

    u_int32_t curr_user_id = 4294967295;

    for (unsigned int i = 0; i < num_rows; i++) {
        u_int32_t index = uts[i].id;

        if (uts[i].user_id != curr_user_id) {
            curr_user_id = uts[i].user_id;
            p_m_ids.clear();
            p_r_ids.clear();
        }

        if (max_hist > p_m_ids.size()) {
            prev_movie_ids[index] = new u_int32_t[p_m_ids.size()];
            prev_ratings[index]   = new float32_t[p_r_ids.size()];
            num_tokens_prev[index] = p_m_ids.size();

            for (unsigned int j = 0; j < p_m_ids.size(); j++) prev_movie_ids[index][j] = p_m_ids[j];
            for (unsigned int j = 0; j < p_r_ids.size(); j++) prev_ratings[index][j]   = p_r_ids[j];
        }
        else {
            prev_movie_ids[index] = new u_int32_t[max_hist];
            prev_ratings[index]   = new float32_t[max_hist];
            num_tokens_prev[index] = max_hist;
            
            for (unsigned int j = 0; j < max_hist; j++) prev_movie_ids[index][j] = p_m_ids[p_m_ids.size()-max_hist+j];
            for (unsigned int j = 0; j < max_hist; j++) prev_ratings[index][j]   = p_r_ids[p_r_ids.size()-max_hist+j];
        }

        p_m_ids.push_back(movie_id[index]);
        p_r_ids.push_back(normalized_ratings[index]);
    }
}