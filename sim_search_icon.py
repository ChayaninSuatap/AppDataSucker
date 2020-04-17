from similarity_search_util import get_topn_using_icon, euclidean, mse, make_category_dict, \
    get_human_testset_cate_from_fn
from global_util import save_pickle, load_pickle
from sim_search_icon_and_sc import check_sim_search_accuracy

if __name__ == '__main__':
    icon_pc_path    = 'sim_search_t/icon_pc/i10_p.obj'
    icon_hpc_path   = 'sim_search_t/icon_hpc/i10_p.obj'
    save_topn_cache_path = 'sim_search_t/topn_cache/i10_p_top5.obj'
    topn = 5

    output = get_topn_using_icon(
        icon_pc_path   = icon_pc_path,
        icon_hpc_path  = icon_hpc_path,
        distance_fn = euclidean,
        topn = topn,
    )
    save_pickle(output, save_topn_cache_path)
    check_sim_search_accuracy(save_topn_cache_path)

