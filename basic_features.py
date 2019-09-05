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
    for app_ids, icons, labels in datagen:
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

def split_train_test(dataset_path, train_path, test_path, k_iter, compress=3):
    #split train test from dataset and also normalize it

    gist_dict = load(dataset_path)
    aial_train , aial_test = dataset_util.prepare_aial_train_test(k_iter)

    gist_train_dict = {}
    gist_test_dict = {}

    for app_id, _, _ in aial_train:
        if app_id in gist_dict : gist_train_dict[app_id] = gist_dict[app_id]

    for app_id, _, _ in aial_test:
        if app_id in gist_dict : gist_test_dict[app_id] = gist_dict[app_id]

    #normalize

    scaler = fit_scaler(gist_train_dict)
    transform_to_scaler(gist_train_dict, scaler)

    scaler = fit_scaler(gist_test_dict)
    transform_to_scaler(gist_test_dict, scaler)

    #add label
    for app_id,_, cate in aial_train:
        if app_id in gist_train_dict : gist_train_dict[app_id] = gist_train_dict[app_id], cate

    for app_id,_, cate in aial_test:
        if app_id in gist_test_dict : gist_test_dict[app_id] = gist_test_dict[app_id], cate

    dump(gist_train_dict, train_path, compress=compress)
    dump(gist_test_dict, test_path, compress=compress)
    print('done')

def make_sc_hog():
    # prepare screenshot k0
    aial_train, aial_test = dataset_util.prepare_aial_train_test(0)
    aial_train = [app_id for app_id,_,_ in aial_train]
    aial_test = [app_id for app_id,_,_ in aial_test]

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

def make_sc_hog_split_train_test(k_iter, compute_train_set=False, compute_test_set=False):
    aial_train, aial_test = dataset_util.prepare_aial_train_test(k_iter)
    if compute_train_set and not compute_test_set: aial = aial_train
    elif compute_test_set and not compute_train_set: aial = aial_test
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
    
    scaler = my_fit_scaler(set_dict)
    my_transform_to_scaler(set_dict, scaler)

    for app_id_fn in set_dict.keys():
        set_dict[app_id_fn] = set_dict[app_id_fn], app_id_cate_dict[app_id_fn[:-6]]

    dump_fn = 'sc_hog_train_k%d.gzip' % (k_iter,) if compute_train_set else 'sc_hog_test_k%d.gzip' % (k_iter,)
    dump(set_dict, 'basic_features/sc_hog/' + dump_fn)

if __name__ == '__main__':
    # make_sc_hog_split_train_test(0, compute_train_set=False, compute_test_set=True)

    # split_train_test('basic_features/icon_gist.gzip', train_path = 'basic_features/icon_gist_train_k3.gzip',
        # test_path = 'basic_features/icon_gist_test_k3.gzip', k_iter = 3)