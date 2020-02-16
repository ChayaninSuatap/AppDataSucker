import os
from global_util import save_pickle, load_pickle
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
from PIL import Image


MAX_CACHE = 20
THRESHOLD = 7500
COMPARE_NEXT_N = 20

__background180 = Image.new('RGBA', (180, 180), (255,255,255))
__background512 = Image.new('RGBA', (512, 512), (255,255,255))

def loadimg(path, cache, img_fn):
    if img_fn in cache:
        return cache[img_fn]
    else:
        png = Image.open(path).convert('RGBA')
        png = png.resize( (180, 180))
        if png.size == (180, 180):
            img = Image.alpha_composite(__background180, png)
        elif png.size == (512, 512):
            img = Image.alpha_composite(__background512, png)
        img = np.array(img)[:,:,:3]

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
    error_image_files_n = 0

    #load saved progress
    # icon_fns = load_pickle('similarity_search/icon_fns.obj')

    #old
    while len(icon_fns) > 0:
        #print progression
        print('progress %.2f' % ((icon_fns_size - len(icon_fns))*100/icon_fns_size,))

        #fetch first icon for main comparer
        comparing_icon_fn = icon_fns[0]
        icon_fns.remove(comparing_icon_fn)
        current_icon_fns = [comparing_icon_fn]

        try:
            comparing_icon = loadimg(path + comparing_icon_fn, cache, comparing_icon_fn)
        except:
            error_image_files_n += 1
            continue
    
        #compare to each other images
        remove_list = []
        compared_n = 0
        for icon_fn in icon_fns:
            try:
                icon_img = loadimg(path + icon_fn, cache, icon_fn)
                if compare(comparing_icon, icon_img, THRESHOLD):
                    current_icon_fns.append(icon_fn)
                    remove_list.append(icon_fn)
                compared_n += 1

                if compared_n == COMPARE_NEXT_N:
                    break

            #simply skip if cant read img
            except Exception as e:
                error_image_files_n += 1
                remove_list.append(icon_fn)
                print('this one is error', icon_fn)
                print(repr(e))

        #copy file for test threshold
        print('copying')

        #copy for duplicate set
        set_dir = 'similarity_search/duplicate_set/' + str(len(current_icon_fns)) + comparing_icon_fn + '/'
        os.mkdir(set_dir)
        for x in current_icon_fns:
            try:
                shutil.copyfile(path + x, set_dir + x)
            except Exception as e:
                pass

        #copy for a unique from set
        shutil.copyfile(path + x, 'similarity_search/icons_remove_duplicate/' + random.choice(current_icon_fns))

        #post process before next loop
        for x in remove_list:
            icon_fns.remove(x)

        #save icon_fns for run at home ..
        save_pickle(icon_fns, 'similarity_search/icon_fns.obj')

        print('done copying. error images n', error_image_files_n)
        print('name of comparer', comparing_icon_fn)



if __name__ == '__main__':
    fn('icons.combine.recrawled/')
