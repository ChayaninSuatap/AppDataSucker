from similarity_search_util import compute_preds, mse, euclidean, make_category_dict, get_human_testset_cate_from_fn
from global_util import save_pickle, load_pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from overall_feature_util import _all_game_category
import random

def get_icon_names(fd):
    return list(os.listdir(fd))

def get_icon_names_filtered(fd, aial_obj = 'aial_seed_327.obj'):
    aial_filtered = load_pickle(aial_obj)

    aial_filtered_dict = {}
    for aial_rec in aial_filtered:
        aial_filtered_dict[aial_rec[0]] =True

    return [fn for fn in list(os.listdir(fd)) if fn[:-4] in aial_filtered_dict]

def map_two_lists(icon_names, preds):
    d = {}
    for icon_name, pred in zip(icon_names, preds):
        d[icon_name] = pred
    return d

def make_preds_cache(icons_fd, model_path, cache_path, use_feature_vector):
    icon_names = get_icon_names_filtered(icons_fd)
    preds = compute_preds(icon_names, model_path = model_path, icons_fd_path = icons_fd, show_output =True, use_feature_vector=use_feature_vector)
    save_pickle(preds, cache_path)

def get_top10_nearest_icon_human_test(saved_icon_fns_preds_dict, model_path,
    distance_fn, compare_icon_fn_list,
    use_feature_vector ,
    save_human_preds_caches_path,
    human_test_fd = 'icons_human_test/',
    load_human_preds_caches_path = None, topn = 10):

    icon_fns_preds_dict = load_pickle(saved_icon_fns_preds_dict)

    print('before filter', len(icon_fns_preds_dict.keys()))
    new_icon_fns_preds_dict = {}
    for icon_fn in compare_icon_fn_list:
        if icon_fn in icon_fns_preds_dict:
            new_icon_fns_preds_dict[icon_fn] = icon_fns_preds_dict[icon_fn]
    icon_fns_preds_dict = new_icon_fns_preds_dict
    print('after filter', len(icon_fns_preds_dict.keys()))

    if load_human_preds_caches_path != None:
        icon_human_preds = load_pickle(load_human_preds_caches_path)
    else:
        icon_human_fns = get_icon_names(human_test_fd)
        icon_human_preds = compute_preds(icon_human_fns, icons_fd_path = human_test_fd, model_path= model_path, use_feature_vector=use_feature_vector)

        save_pickle(icon_human_preds, save_human_preds_caches_path)
        print('saved human preds caches')

    output = {}
    for human_icon_fn, human_icon_pred in icon_human_preds.items():
        icon_fns_distances_pairs = []
        for icon_fn, icon_pred in icon_fns_preds_dict.items():
            icon_fns_distances_pairs.append( (icon_fn, distance_fn(human_icon_pred, icon_pred)))
        
        #sort
        sorted_icon_fns_distances_pairs = sorted(icon_fns_distances_pairs, key=lambda x : x[1])[:topn]
        print(human_icon_fn)
        [print(x[1], end = ' ') for x in sorted_icon_fns_distances_pairs]
        print()
        top10_nearest_icon_fns = [x[0] for x in sorted_icon_fns_distances_pairs]
        output[human_icon_fn] = top10_nearest_icon_fns
    
    return output
    
if __name__ == '__main__':
    # precompute
    # make_preds_cache(icons_fd = 'similarity_search/icons_remove_duplicate/', model_path = 'similarity_search/models/cate_model5_fix_cw_k0.hdf5',
    # cache_path='similarity_search/icon_names_preds_dict_model5_fix_cw_k0_softmax.obj', use_feature_vector=False)

    # compare_icon_fn_list = get_icon_names('similarity_search/icons_remove_duplicate')
    # output = get_top10_nearest_icon_human_test('similarity_search/icon_names_preds_dict_model5_fix_cw_k0.obj',compare_icon_fn_list = compare_icon_fn_list,
    #     model_path = 'similarity_search/models/cate_model5_fix_cw_k0.hdf5', distance_fn=euclidean, use_feature_vector=True)
    # save_pickle(output, 'output.obj')
    # input('done')


    '''
    cate_model5_fix_cw_k0-ep-2120-loss-0.004-acc-0.999-vloss-5.843-vacc-0.379.hdf5
cate_model_i2_cw_k0-ep-101-loss-0.077-acc-0.975-vloss-0.399-vacc-0.375.hdf5
cate_model_i3_cw_k0-ep-1853-loss-0.003-acc-0.999-vloss-5.855-vacc-0.362.hdf5
cate_model_i5_cw_k0-ep-2271-loss-0.010-acc-0.997-vloss-4.939-vacc-0.352.hdf5
cate_model_i7_cw_k0-ep-1037-loss-0.020-acc-0.998-vloss-6.140-vacc-0.329.hdf5
    '''

    cate_dict = make_category_dict()
 
    output = load_pickle('output.obj') #human testset icon fn -> list of nearest fn (top ten, already sorted)

    fig, ax = plt.subplots(5, 6, figsize=(8,4))    
    [x.set_axis_off() for x in ax.ravel()]

    for i in range(5):
        current_key = random.choice(list(output.keys())) #sample a human testset icon fn
        ax[i,0].imshow(Image.open('icons_human_test/' + current_key))
        ax[i,0].text(0, 0.5, get_human_testset_cate_from_fn(current_key))
        for j in range(1,6):
            ax[i,j].imshow(Image.open('icons.backup/' + output[current_key][j]))
            ax[i,j].text(0, 0.5, '%s \n%s' % (cate_dict[output[current_key][j][:-4]], output[current_key][j][:-4]))
    plt.show()







    



