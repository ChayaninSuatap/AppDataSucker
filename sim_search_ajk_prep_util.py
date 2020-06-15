from global_util import load_pickle, save_pickle
import os
import keras_util
from shutil import copyfile
import numpy as np

def fn(k, source_icon_fd, dest_icon_fd, cate_index, aial_obj = 'aial_seed_327.obj'):
    aial = load_pickle(aial_obj)
    _, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k, n_splits=4)
    app_ids = [x[0] for x in aial_test if np.array(x[2]).argmax() == cate_index]

    for app_id in app_ids:
        copyfile(source_icon_fd + app_id + '.png', dest_icon_fd + app_id + '.png')

if __name__ == '__main__':
    source_icon_fd = 'icons.combine.recrawled/'
    dest_icon_fd = 'sim_search_ajk/icons_k1_sports/'
    fn(k = 1, source_icon_fd = source_icon_fd , dest_icon_fd = dest_icon_fd, cate_index = 7)
