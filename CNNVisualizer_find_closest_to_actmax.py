import os
import numpy as np
from PIL import Image
from icon_util import load_icon_by_fn, rgb_to_gray
import matplotlib.pyplot as plt
from global_util import save_pickle, load_pickle

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

def make_cache():
    fd_actmax = 'visualize_cnn/activation_maximization_amw2_iter1000/'
    fd_training = 'visualize_cnn/training_set_by_genre/'

    cache = {}

    for cate_fd in os.listdir(fd_training):
        cate = cate_fd
        cate_fd += '/'
        actmax_img = load_icon_by_fn(fd_actmax + cate + '.png', 128, 128)/ 255
        cache[cate] = []
        for icon_fn in os.listdir(fd_training + cate_fd):
            icon = load_icon_by_fn(fd_training + cate_fd + icon_fn, 128, 128)/255
            diff = actmax_img - icon
            diff = np.absolute(diff)
            diff = diff.sum()
            cache[cate].append((icon_fn, diff))
    
    save_pickle(cache, 'icon_diff_actmax.obj')

if __name__ == '__main__':
    fd_actmax = 'visualize_cnn/activation_maximization_amw2_iter1000/'
    fd_training = 'visualize_cnn/training_set_by_genre/'

    cache = load_pickle('icon_diff_actmax.obj')
    for cate in cates:
        print(cate)
        sorted_cached_by_cate = sorted(cache[cate], key=lambda x : x[1])
        for icon_fn, diff in (sorted_cached_by_cate[:5]):
            print(icon_fn, diff)
            # Image.open(fd_training + cate + '/' + icon_fn).show()
            os.startfile('D:/crawler/'+fd_training + cate + '/' + icon_fn)
            input()
    


