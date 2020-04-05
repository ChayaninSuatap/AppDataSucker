from similarity_search import make_preds_cache

if __name__ == '__main__':

    icons_fd = 'similarity_search/icons_rem_dup_human_recrawl/'
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    cache_path = 'sim_search_t/preds_caches/icon_model2.4_k3_t.obj'
    use_feature_vector = False
    human_test_fd = 'icons_human_test/'

    # make_preds_cache(
    #     icons_fd = icons_fd,
    #     model_path = model_path,
    #     cache_path = cache_path,
    #     use_feature_vector = use_feature_vector)

    