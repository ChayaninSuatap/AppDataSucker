from similarity_search import make_preds_cache, \
    get_top10_nearest_icon_human_test, get_icon_names_filtered

from similarity_search_util import euclidean, mse, \
    get_human_testset_cate_from_fn, make_category_dict, create_mean_preds_caches

from global_util import save_pickle, load_pickle
import random
from PIL import Image
import matplotlib.pyplot as plt
import os

def plot_human_icon_top10(human_icon_top10_cache_fn,
    icons_fd,
    human_icon_fd = 'icons_human_test/'):
    cate_dict = make_category_dict()
 
    output = load_pickle(human_icon_top10_cache_fn) #human testset icon fn -> list of nearest fn (top ten, already sorted)

    fig, ax = plt.subplots(5, 6, figsize=(8,4))    
    [x.set_axis_off() for x in ax.ravel()]

    for i in range(5):
        current_key = random.choice(list(output.keys())) #sample a human testset icon fn
        ax[i,0].imshow(Image.open(human_icon_fd + current_key))
        ax[i,0].text(0, 0.5, get_human_testset_cate_from_fn(current_key))
        for j in range(1,6):
            ax[i,j].imshow(Image.open(icons_fd + output[current_key][j-1]))
            print(cate_dict[output[current_key][j-1][:-4]])
            print(output[current_key][j-1][:-4])
            ax[i,j].text(0, 0.5, '%s \n%s' % (cate_dict[output[current_key][j-1][:-4]], output[current_key][j-1][:-4]))
    plt.show()

def check_sim_search_accuracy(human_icon_top10_cache_fn):
    cate_dict = make_category_dict()

    output = load_pickle(human_icon_top10_cache_fn) #human testset icon fn -> list of nearest fn (top ten, already sorted)

    correct = 0

    for human_icon_app_id, top10_app_ids in output.items():
        answer = get_human_testset_cate_from_fn(human_icon_app_id)

        for top10_app_id in top10_app_ids:
            if answer == cate_dict[top10_app_id[:-4]]:
                correct += 1
    print('total corrects', correct)
    print('total sim predicted', len(output) * len(top10_app_ids))
    print('acc', correct / (len(output) * len(top10_app_ids)))


if __name__ == '__main__':

    icon_projs = [
        # 'icon_model2.4',
        # 'icon_model2.3',
        # 'icon_model1.2',
        # 'icon_model1.3',
        'icon_model1.5',
    ]

    for icon_proj in icon_projs:

        for model_fn in os.listdir('ensemble_models_t/' + icon_proj):

            if 'k0' in model_fn: k=0
            if 'k1' in model_fn: k=1
            if 'k2' in model_fn: k=2
            if 'k3' in model_fn: k=3

            proj_name =  icon_proj + '_k' + str(k) + '_p'
            use_feature_vector = False

            #is overrided
            # model_fn = 'icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'

            icons_fd = 'similarity_search/icons_rem_dup_human_recrawl/'
            model_path = 'ensemble_models_t/' + icon_proj + '/' + model_fn
            cache_path = 'sim_search_t/preds_caches/%s.obj' % (proj_name,)
            human_test_fd = 'icons_human_test/'
            human_preds_caches_fd = 'sim_search_t/human_preds_caches/' 
            human_preds_caches_path = human_preds_caches_fd + proj_name + '.obj'
            human_icon_top10_cache_fd = 'sim_search_t/human_icon_top10_cache/'
            human_icon_top10_cache_path = human_icon_top10_cache_fd + proj_name + '.obj'

            make_preds_cache(
                icons_fd = icons_fd,
                model_path = model_path,
                cache_path = cache_path,
                use_feature_vector = use_feature_vector)

            human_icon_top10_cache = get_top10_nearest_icon_human_test(cache_path,
                model_path, euclidean,
                get_icon_names_filtered(icons_fd),
                use_feature_vector,
                human_preds_caches_path,
                load_human_preds_caches_path = None)

            save_pickle(human_icon_top10_cache,
                human_icon_top10_cache_fd  + proj_name + '.obj')
            check_sim_search_accuracy(human_icon_top10_cache_path)


            # create_mean_preds_caches('sim_search_t/human_preds_caches/', 
            #     [
            #         'icon_model2.4_k0_t_p.obj',
            #         'icon_model2.4_k1_t_p.obj',
            #         'icon_model2.4_k2_t_p.obj',
            #         'icon_model2.4_k3_t_p.obj'
            #     ],
            #     'sim_search_t/human_preds_caches/icon_model2.4_p_e.obj')

