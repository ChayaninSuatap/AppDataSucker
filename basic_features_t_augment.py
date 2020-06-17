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


def split_aial(aial_obj, k_iter):
    aial_train, aial_test = keras_util.gen_k_fold_pass(aial_obj, kf_pass = k_iter, n_splits=4)
    aial_train_new, aial_test_new = [], []
    for app_id, _, cate, *_ in aial_train:
        aial_train_new.append( (app_id, cate))
    for app_id, _, cate, *_ in aial_test:
        aial_test_new.append( (app_id,cate))
    return aial_train_new, aial_test_new

def x_generator(aial_train, batch_size, samples_fd, resize_w, resize_h, extract_fn, rotate_for_sc=False):
    datagen = keras_util.create_image_data_gen()
    imgs_now = []
    cates_now = []
    while True:
        random.shuffle(aial_train)
        for app_id, cate in aial_train:
            img = load_icon_by_fn(samples_fd + app_id + '.png', resizeW = resize_w, resizeH = resize_h, rotate_for_sc = rotate_for_sc)
            imgs_now.append(img)
            cates_now.append(cate)

            #augment entire batch
            if len(imgs_now) == batch_size:
                for chrunk_gen in datagen.flow(np.array(imgs_now), batch_size = batch_size, shuffle=False):
                    break

                features = []
                for img_in_chrunk in chrunk_gen:
                    feature = extract_fn(img_in_chrunk/255)
                    features.append(feature)
                yield np.array(features), np.array(cates_now)

                imgs_now = []
                cates_now = []
                
def make_test_set(aial_test, samples_fd,  resize_w, resize_h, extract_fn, rotate_for_sc=False):
    features = []
    cates = []
    for app_id, cate in aial_test:
        img = load_icon_by_fn(samples_fd + app_id + '.png', resizeW=resize_w, resizeH=resize_h, rotate_for_sc=rotate_for_sc)
        feature = extract_fn(img/255)
        features.append(feature)
        cates.append(cate)
    return np.array(features), np.array(cates)

if __name__ == '__main__':
    #setting
    k_iter = 0
    samples_fd = 'icons.combine.recrawled/'
    epochs = 100
    batch_size = 32
    resize_w = 180
    resize_h = 180
    
    def extract_fn(img):
        return extract_hog(img, pixels_per_cell=(8,8))
        
    #code
    aial_obj = load_pickle('aial_seed_327.obj')
    aial_train, aial_test = split_aial(aial_obj, k_iter)

    x_gen = x_generator(aial_train, batch_size, samples_fd=samples_fd, resize_w=resize_w, resize_h=resize_h, extract_fn=extract_fn)
    test_set = make_test_set(aial_test, samples_fd, resize_w, resize_h, extract_fn)

    model = make_model('hog')
    model.fit_generator(x_gen, steps_per_epoch = math.ceil(len(aial_train)/batch_size),
        epochs=epochs, validation_data=test_set)
