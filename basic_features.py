import cv2
import random
import numpy as np
import preprocess_util
import keras_util
import icon_cate_util
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.color import rgb2gray
from joblib import dump, load
import global_util
import dataset_util
from tensorflow.keras.layers import Dense, Conv2D, Input, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import os
import sc_util
import mypath
import math

def _load_aial():
    random.seed(859)
    np.random.seed(859)
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    random.shuffle(aial)
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    return aial

def _icon_generator():
    aial = _load_aial()
    datagen = icon_cate_util.datagenerator(aial, 32, 1,  cate_only=True, yield_app_id=True, icon_resize_dim=(180,180))
    for app_ids, icons, _ in datagen:
        for app_id, icon in zip(app_ids, icons):
            yield app_id, icon

def _screenshot_generator(batch_size = 256, skip_reading_image=False, skip_reading_amount=0):
    mypath.screenshot_folder = 'screenshots.256.distincted/'
    sc_dict = sc_util.make_sc_dict()
    aial = _load_aial()
    aial, _ = sc_util.make_aial_sc(aial, [], sc_dict)
    datagen = icon_cate_util.datagenerator(aial, batch_size = batch_size, epochs = 1, cate_only=True, train_sc=True, yield_app_id=True
        ,skip_reading_image=skip_reading_image, skip_reading_amount=skip_reading_amount)

    for app_ids, scs, _ in datagen:
        for app_id, sc in zip(app_ids, scs):
            yield app_id, sc

def fit_scaler(feature_dict):
    l = []
    for k,item in feature_dict.items():
        l.append(item)
    scaler = StandardScaler()
    scaler.fit(l)
    del l
    return scaler

def transform_to_scaler(feature_dict, scaler):
    for k, item in feature_dict.items():
        feature_dict[k] = (item-scaler.mean_) / scaler.var_

def my_fit_scaler(feature_dict:dict):
    sum = None
    for k, item in feature_dict.items():
        if sum is None:
            sum = np.array(item)
            np_sum = [item]
        else:
            sum+=item
            np_sum.append(item)

    mean = sum / len(feature_dict)

    sum_diff = None
    for item in feature_dict.values():
        if sum_diff is None:
            sum_diff = (item - mean) ** 2
        else:
            sum_diff += (item - mean) ** 2
    
    var = sum_diff / (len(feature_dict))

    return mean, var

def my_transform_to_scaler(feature_dict, mean, var):
    for k, item in feature_dict.items():
        feature_dict[k] = (item - mean) / var

def make_model(feature, dense_sizes=[1000,500]):

    if feature == 'hog':
        input_shape = 32400
    elif feature == 'gist':
        input_shape = 512
    
    def add_dense(size, last_layer):
        x = Dense(size)(last_layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x
    
    input_layer = Input(shape=(input_shape,))
    x = add_dense(dense_sizes[0], input_layer)

    for i in range(1, len(dense_sizes)):
        x = add_dense(dense_sizes[i], x)

    x = Dense(17, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

def load_dataset(train_path, test_path):
    xtrain, xtest, ytrain, ytest = [],[],[],[]

    feature_dict_train = load(train_path)
    for k, (feature, label) in feature_dict_train.items():
        xtrain.append( feature)
        ytrain.append( label)
    del feature_dict_train

    feature_dict_test = load(test_path)
    for k, (feature, label) in feature_dict_test.items():
        xtest.append( feature)
        ytest.append( label)
    del feature_dict_test

    return np.array(xtrain), np.array(xtest), np.array(ytrain), np.array(ytest)

def extract_hog(icon, pixels_per_cell=(8,8)):
    denorm_icon = (icon*255).astype('int32')
    hog_feature = hog(denorm_icon, visualize=False, multichannel=True, pixels_per_cell=pixels_per_cell)
    return hog_feature

def extract_hog_sc(sc):
    return extract_hog(sc, pixels_per_cell=(10,10))

def split_train_test(dataset_path, train_path, test_path, k_iter, compress=3, sc=False):
    #split train test from dataset and also normalize it
    gist_dict = load(dataset_path)

    if sc:
        mypath.screenshot_folder = 'screenshots.256.distincted/'
        sc_dict = sc_util.make_sc_dict()
        aial_train , aial_test = dataset_util.prepare_aial_train_test(k_iter)
        aial_train, aial_test = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)
    else:
        aial_train , aial_test = dataset_util.prepare_aial_train_test(k_iter)

    gist_train_dict = {}
    gist_test_dict = {}

    # aial train & test need cate !
    for app_id,_,_ in aial_train:
        if app_id in gist_dict : gist_train_dict[app_id] = gist_dict[app_id]

    for app_id,_,_ in aial_test:
        if app_id in gist_dict : gist_test_dict[app_id] = gist_dict[app_id]

    #normalize

    scaler = fit_scaler(gist_train_dict)
    transform_to_scaler(gist_train_dict, scaler)

    transform_to_scaler(gist_test_dict, scaler)

    #add label
    for app_id,_, cate in aial_train:
        if app_id in gist_train_dict : gist_train_dict[app_id] = gist_train_dict[app_id], cate

    for app_id,_, cate in aial_test:
        if app_id in gist_test_dict : gist_test_dict[app_id] = gist_test_dict[app_id], cate

    dump(gist_train_dict, train_path, compress=compress)
    dump(gist_test_dict, test_path, compress=compress)
    print('done')

def make_icon_hog(pixels_per_cell=(8,8)):
    dict = {}
    for app_id, icon in _icon_generator():
        print(app_id)
        dict[app_id] = extract_hog(icon, pixels_per_cell=pixels_per_cell)
    dump(dict, 'basic_features/icon_hog16.gzip', compress=3)

def make_sc_hog():
    # total sc 141895
    feature_num_in_part = 0
    file_part_i = 0
    max_num_in_part = 14000

    dict = {}
    for app_id, sc in _screenshot_generator():
        print(file_part_i, feature_num_in_part,'/', max_num_in_part)
        feature = extract_hog_sc(sc)
        dict[app_id] = feature
        feature_num_in_part += 1

        if feature_num_in_part == max_num_in_part:
            dump(dict, 'basic_features/sc_hog%02d.gzip' % (file_part_i,), compress=3)
            dict = {}
            feature_num_in_part = 0
            file_part_i += 1
    
    if feature_num_in_part > 0 :
        dump(dict, 'basic_features/sc_hog%02d.gzip' % (file_part_i,), compress=3)

def make_sc_hog_split_for_generator(feature_dict_path, save_split_dir, k_iter, train=False, test=False):
    if train == test:
        raise Exception('select train or test')

    feature_dict = load(feature_dict_path)
    max_num_in_part = 10000
    feature_num_in_part = 0
    file_part_i = 0

    dict = {}
    for app_id, feature in feature_dict.items():
        dict[app_id] = feature
        feature_num_in_part += 1

        if feature_num_in_part == max_num_in_part:
            if train:
                dump(dict, '%s/sc_hog_train_split%02d_k%d' % (save_split_dir,file_part_i, k_iter), compress=0)
            else:
                dump(dict, '%s/sc_hog_test_split%02d_k%d' % (save_split_dir,file_part_i, k_iter), compress=0)
            dict = {}
            feature_num_in_part = 0
            print('part', file_part_i, 'finished')
            file_part_i += 1
    
    if feature_num_in_part > 0 :
        if train:
            dump(dict, '%s/sc_hog_train_split%02d_k%d' % (save_split_dir,file_part_i, k_iter), compress=0)
        else:
            dump(dict, '%s/sc_hog_test_split%02d_k%d' % (save_split_dir,file_part_i, k_iter), compress=0)

    print('done')

#WIP
def make_sc_hog_generator(epochs, batch_size, set_path):
    #note
    #random each epoch
    for set_fn in os.listdir(set_path):
        pass

def make_sc_hog_split_train_test(k_iter, compute_train_set=False, compute_test_set=False, mean=None, var=None, compress=3):
    aial_train, aial_test = dataset_util.prepare_aial_train_test(k_iter)
    if compute_train_set and not compute_test_set:
        aial = aial_train
    elif compute_test_set and not compute_train_set:
        if mean is None or var is None:
            raise Exception('Give me mean or var when making test set')
        aial = aial_test
    else:
        raise Exception('Choose only one : train or test?')

    looking_app_ids = [app_id for app_id,_,_ in aial]

    app_id_cate_dict = {}
    for app_id,_,cate in aial:
        app_id_cate_dict[app_id] = cate

    set_dict = {}
    for featurefn in os.listdir('basic_features/sc_hog'):
        print('seaching in', featurefn)
        feature_dict = load('basic_features/sc_hog/' + featurefn)

        for app_id in feature_dict.keys():
            if app_id[:-6] in looking_app_ids:
                set_dict[app_id] = feature_dict[app_id]
        
        del feature_dict
    
    if compute_train_set:
        mean, var = my_fit_scaler(set_dict)
    my_transform_to_scaler(set_dict, mean, var)

    for app_id_fn in set_dict.keys():
        set_dict[app_id_fn] = set_dict[app_id_fn], app_id_cate_dict[app_id_fn[:-6]]

    dump_fn = 'sc_hog_train_k%d.gzip' % (k_iter,) if compute_train_set else 'sc_hog_test_k%d.gzip' % (k_iter,)
    dump(set_dict, 'basic_features/sc_hog/' + dump_fn, compress=compress)

    return mean, var

if __name__ == '__main__':

    #split gist train test
    split_train_test('basic_features/sc_gist.gzip', train_path = 'basic_features/sc_gist_train_k0.gzip',
        test_path = 'basic_features/sc_gist_test_k0.gzip', k_iter = 0, sc=True)
    
    #make sc_hog test set
    # mean, var = global_util.load_pickle('basic_features/sc_hog_mean_and_var_k0.obj')
    # make_sc_hog_split_train_test(0, compute_test_set=True, mean=mean, var=var)

    # compare mean var
    # feature_dict = load('basic_features/icon_gist.gzip')
    # aial_train, aial_test = dataset_util.prepare_aial_train_test(0, True)
    # train = {}
    # test = {}
    # for k, item in feature_dict.items():
    #     if k in aial_train:
    #         train[k] = item
    #     elif k in aial_test:
    #         test[k] = item
    #     else:
    #         raise Exception('FUCKED')
    # train_mean, train_var = my_fit_scaler(train) 
    # test_mean, test_var = my_fit_scaler(test)

    # print(train_mean[:5], train_var[:5])
    # print(test_mean[:5], test_var[:5])
