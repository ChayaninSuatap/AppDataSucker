import os
from global_util import save_pickle, load_pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random

MAX_CACHE = 9000
THRESHOLD = 10000

def loadimg(path, cache, img_fn):
    if img_fn in cache:
        return cache[img_fn]
    else:
        img = plt.imread(path)

        #remove an image if exceed
        if len(cache.keys()) > MAX_CACHE:
            cache.pop( random.choice(list( cache.keys())))
        
        cache[img_fn] = img
    return img

def compare(img_a, img_b, threshold):
    diff = np.sum(np.absolute(img_a[:, :, :3] - img_b[:, :, :3]))
    return diff < threshold

def fn(path='icons.backup/'):
    cache = {}
    icon_fns = list(os.listdir(path))
    icon_fns_size = len(icon_fns)

    #load saved progress
    icon_fns = load_pickle('similarity_search/icon_fns.obj')

    while len(icon_fns) > 0:
        #print progression
        print('progress %.2f' % ((icon_fns_size - len(icon_fns))*100/icon_fns_size,))

        #fetch first icon for main comparer
        comparing_icon_fn = icon_fns[0]
        icon_fns.remove(comparing_icon_fn)
        current_icon_fns = [comparing_icon_fn]
        try:
            comparing_icon = loadimg(path + comparing_icon_fn, cache, comparing_icon_fn)
        except Exception as e:
            continue
    
        #compare to each other images
        remove_list = []
        for icon_fn in icon_fns:
            try:
                icon_img = loadimg(path + icon_fn, cache, icon_fn)
                if compare(comparing_icon, icon_img, THRESHOLD):
                    current_icon_fns.append(icon_fn)
                    remove_list.append(icon_fn)
            except Exception as e:
                remove_list.append(icon_fn)

        #copy file for test threshold
        print('copying')
        set_dir = 'similarity_search/duplicate_set/' + str(len(current_icon_fns)) + comparing_icon_fn + '/'
        os.mkdir(set_dir)
        for x in current_icon_fns:
            shutil.copyfile(path + x, set_dir + x)
            shutil.copyfile(path + x, 'similarity_search/icons_remove_duplicate/' + x)

        #post process before next loop
        for x in remove_list:
            icon_fns.remove(x)

        #save icon_fns for run at home ..
        save_pickle(icon_fns, 'similarity_search/icon_fns.obj')

        print('done copying')
        print('name of comparer', comparing_icon_fn)



if __name__ == '__main__':
    fn()
