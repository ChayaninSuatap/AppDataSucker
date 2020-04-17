from similarity_search_util import get_topn_using_screenshot, euclidean, mse, make_category_dict, \
    get_human_testset_cate_from_fn
from global_util import save_pickle, load_pickle
from sim_search_icon_and_sc import check_sim_search_accuracy

if __name__ == '__main__':
    sc_pc_path    = 'sim_search_t/sc_pc/s9_f_e.obj'
    sc_hpc_path   = 'sim_search_t/sc_hpc/s9_f_e_only_origin.obj'
    save_topn_cache_path = 'sim_search_t/topn_cache/s9_f_top10.obj'
    topn = 10

    output = get_topn_using_screenshot(
        sc_pc_path   = sc_pc_path,
        sc_hpc_path  = sc_hpc_path,
        distance_fn = euclidean,
        topn = topn,
    )
    save_pickle(output, save_topn_cache_path)
    check_sim_search_accuracy(save_topn_cache_path)

