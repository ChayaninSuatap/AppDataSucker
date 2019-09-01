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

def fit_scaler(feature_dict):
    l = []
    for k,item in feature_dict.items():
        l.append(item)
    scaler = StandardScaler()
    scaler.fit(l)
    return scaler

def transform_to_scaler(feature_dict, scaler):
    for k, item in feature_dict.items():
        feature_dict[k] = (item-scaler.mean_) / scaler.var_

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

def extract_hog(icon):
    denorm_icon = (icon*255).astype('int32')
    hog_feature = hog(denorm_icon, visualize=False, multichannel=True)
    return hog_feature

if __name__ == '__main__':

    feature_dict = load('basic_features/icon_hog.gzip')
    feature_dict_train = {}
    feature_dict_test = {}

    aial_train, aial_test = dataset_util.prepare_aial_train_test(0)

    app_ids_train = [app_id for app_id,_,_ in aial_train]
    app_ids_test = [app_id for app_id,_,_ in aial_test]

    for app_id, item in feature_dict.items():
        if app_id in app_ids_train:
            feature_dict_train[app_id] = item
        elif app_id in app_ids_test:
            feature_dict_test[app_id] = item

    print(len(feature_dict.keys()), len(feature_dict_train.keys()), len(feature_dict_test.keys()))
    
    dump(feature_dict_train, 'basic_features/icon_hog_train_k0.gzip', compress=3)
    dump(feature_dict_test, 'basic_features/icon_hog_test_k0.gzip', compress=3)


    

