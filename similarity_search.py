from similarity_search_util import compute_preds, mse, euclidean, make_category_dict, get_human_testset_cate_from_fn
from global_util import save_pickle, load_pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from overall_feature_util import _all_game_category
import random

def get_icon_names(fd):
    return list(os.listdir(fd))

def map_two_lists(icon_names, preds):
    d = {}
    for icon_name, pred in zip(icon_names, preds):
        d[icon_name] = pred
    return d

def make_preds_cache():
    icon_names = get_icon_names('icons.backup')
    preds = compute_preds(icon_names, icons_fd_path = 'icons.backup/', show_output=True)
    save_pickle(preds, 'similarity_search/icon_names_preds_dict.obj')

def get_top10_nearest_icon_human_test(distance_fn = mse):
    saved_icon_fns_preds_dict = 'similarity_search/icon_names_preds_dict.obj' # list of list : [[0.34, 0.12 , 0.677]]
    human_test_fd = 'icons_human_test/'

    icon_fns_preds_dict = load_pickle(saved_icon_fns_preds_dict)
    icon_human_fns = get_icon_names(human_test_fd)
    icon_human_preds = compute_preds(icon_human_fns, icons_fd_path = human_test_fd)

    output = {}
    for human_icon_fn, human_icon_pred in icon_human_preds.items():
        icon_fns_distances_pairs = []
        for icon_fn, icon_pred in icon_fns_preds_dict.items():
            icon_fns_distances_pairs.append( (icon_fn, distance_fn(human_icon_pred, icon_pred)))
        
        #sort
        sorted_icon_fns_distances_pairs = sorted(icon_fns_distances_pairs, key=lambda x : x[1])[:10]
        top10_nearest_icon_fns = [x[0] for x in sorted_icon_fns_distances_pairs]
        output[human_icon_fn] = top10_nearest_icon_fns
    
    return output
    
if __name__ == '__main__':
    # precompute
    # output = get_top10_nearest_icon_human_test()
    # save_pickle(output, 'output.obj')

    cate_dict = make_category_dict()

    output = load_pickle('output.obj') #human testset icon fn -> list of nearest fn

    fig, ax = plt.subplots(5, 6, figsize=(8,4))    
    [x.set_axis_off() for x in ax.ravel()]

    for i in range(5):
        current_key = random.choice(list(output.keys())) #sample a human testset icon fn


        ax[i,0].imshow(Image.open('icons_human_test/' + current_key))
        ax[i,0].text(0, 0.5, get_human_testset_cate_from_fn(current_key))
        for j in range(1,6):
            ax[i,j].imshow(Image.open('icons.backup/' + output[current_key][j]))
            ax[i,j].text(0, 0.5, '%s' % (cate_dict[output[current_key][j][:-4]],))
    plt.show()







    



