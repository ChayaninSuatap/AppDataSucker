from shutil import copyfile
import os
from global_util import load_pickle, save_pickle
import keras_util
from basic_features import make_model, extract_hog
from icon_util import load_icon_by_fn
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import multiprocessing

def split_aial(aial_obj, k_iter):
    aial_train, aial_test = keras_util.gen_k_fold_pass(aial_obj, kf_pass = k_iter, n_splits=4)
    aial_train_new, aial_test_new = [], []
    for app_id, _, cate, *_ in aial_train:
        aial_train_new.append( (app_id, cate))
    for app_id, _, cate, *_ in aial_test:
        aial_test_new.append( (app_id,cate))
    return aial_train_new, aial_test_new

def x_generator(aial_train, batch_size, samples_fd, resize_w, resize_h, pool,
    rotate_for_sc=False, parallel=True, time_extracting=False):
    datagen = keras_util.create_image_data_gen()
    imgs_now = []
    cates_now = []

    global extract_fn
    #make cache
    cache_d = {}
    print('making cache')
    for app_id, cate in aial_train:
        cache_d[app_id] = load_icon_by_fn(samples_fd + app_id + '.png', resizeW = resize_w, resizeH = resize_h, rotate_for_sc = rotate_for_sc)
    print('making cache done')

    while True:
        random.shuffle(aial_train)
        for app_id, cate in aial_train:
            img = cache_d[app_id]
            imgs_now.append(img)
            cates_now.append(cate)

            #augment entire batch
            if len(imgs_now) == batch_size:
                for chrunk_gen in datagen.flow(np.array(imgs_now), batch_size = batch_size, shuffle=False):
                    break

                if time_extracting:
                    start_time = time.time()
                if parallel:
                    features = list(pool.map(extract_fn, chrunk_gen))
                else:
                    features = []
                    for img_in_chrunk in chrunk_gen:
                        feature = extract_fn(img_in_chrunk)
                        features.append(feature)
                if time_extracting:
                    print('extract time per batch', time.time() - start_time)
                yield np.array(features), np.array(cates_now)

                imgs_now = []
                cates_now = []
                
def make_test_set(aial_test, samples_fd,  resize_w, resize_h, rotate_for_sc=False):
    global extract_fn
    features = []
    cates = []
    for app_id, cate in aial_test:
        img = load_icon_by_fn(samples_fd + app_id + '.png', resizeW=resize_w, resizeH=resize_h, rotate_for_sc=rotate_for_sc)
        feature = extract_fn(img)
        features.append(feature)
        cates.append(cate)
    return np.array(features), np.array(cates)

def make_global_extract_fn(fn):
    ##WARNING MUST ACCESS TO TOP LEVEL FOR POOL
    global extract_fn
    extract_fn = fn

if __name__ == '__main__':

    def fn(img):
        feature = extract_hog(img/255, pixels_per_cell=(16,16))
        return feature
    make_global_extract_fn(fn)

    #setting
    k_iter = 0
    samples_fd = 'icons.combine.recrawled/'
    epochs = 100
    batch_size = 256
    resize_w = 180
    resize_h = 180
        
    #code
    pool = multiprocessing.Pool()
    aial_obj = load_pickle('aial_seed_327.obj')
    aial_train, aial_test = split_aial(aial_obj, k_iter)

    x_gen = x_generator(aial_train, batch_size, samples_fd=samples_fd, resize_w=resize_w,
        resize_h=resize_h, pool=pool, parallel=False, time_extracting=True)
    test_set = make_test_set(aial_test, samples_fd, resize_w, resize_h)

    model = make_model('hog', input_shape=6561)
    model.fit_generator(x_gen, steps_per_epoch = math.ceil(len(aial_train)/batch_size),
        # validation_data = test_set,
        epochs=epochs)
