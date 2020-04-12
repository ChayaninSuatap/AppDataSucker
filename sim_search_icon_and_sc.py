from similarity_search_util import get_topn_using_icon_and_screenshot, euclidean, mse, make_category_dict, \
    get_human_testset_cate_from_fn
from global_util import save_pickle, load_pickle

def check_sim_search_accuracy(topn_cache_path):
    cate_dict = make_category_dict()
    output = load_pickle(topn_cache_path) #human testset icon fn -> list of nearest fn (top ten, already sorted)
    correct = 0

    for human_icon_app_id, top10_app_ids in output.items():
        answer = get_human_testset_cate_from_fn(human_icon_app_id)

        for top10_app_id in top10_app_ids:
            if top10_app_id[0][:-4] in cate_dict:
                if answer == cate_dict[top10_app_id[0][:-4]]:
                    correct += 1
            else:
                if answer == cate_dict[top10_app_id[0][:-6]]:
                    correct += 1

    print('total corrects', correct)
    print('total sim predicted', len(output) * len(top10_app_ids))
    print('acc', correct / (len(output) * len(top10_app_ids)))

if __name__ == '__main__':
    icon_pc_path  = 'sim_search_t/icon_pc/icon_top5_p_e.obj'
    icon_hpc_path = 'sim_search_t/icon_hpc/icon_top5_p_e.obj'
    sc_pc_path    = 'sim_search_t/sc_pc/s9_p.obj'
    sc_hpc_path   = 'sim_search_t/sc_hpc/s9_p.obj'
    save_topn_cache_path = 'sim_search_t/topn_cache/s9_p_top5.obj'
    topn = 5

    #VERY IMPORTANT!
    bypass_icon = True

    output = get_topn_using_icon_and_screenshot(
        icon_pc_path = icon_pc_path,
        icon_hpc_path=icon_hpc_path,
        sc_pc_path   = sc_pc_path,
        sc_hpc_path  = sc_hpc_path,
        distance_fn = euclidean,
        topn = topn,
        bypass_icon=bypass_icon
    )
    save_pickle(output, save_topn_cache_path)

    check_sim_search_accuracy(save_topn_cache_path)

