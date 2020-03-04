import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass, metric_top_k, eval_top_5, compute_class_weight, SaveBestEpochCallback
from tensorflow.keras.models import load_model
import keras
import functools
import icon_cate_data_export
import global_util
import mypath
import keras_util
import matplotlib.pyplot as plt
import sc_util
from my_fit_generator import my_fit_generator

batch_size = 32
epochs = 500
random.seed(327)
np.random.seed(327)
k = 0

aial = icon_cate_util.make_aial_from_seed(327, mypath.icon_folder)
aial = icon_cate_util.filter_aial_rating_cate(aial)
aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k, n_splits=4)

cw = icon_cate_util.compute_class_weight_for_cate(aial_train)

#icon
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=True, datagen=keras_util.create_image_data_gen())
# gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=True)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False, enable_cache=True)

model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False,
                                              layers_filters = [64, 128, 256], sliding_dropout = (0, 0.1))

def make_gen_train():
    return icon_cate_util.datagenerator(aial_train, batch_size, epochs=1,cate_only=True, enable_cache=False, datagen=keras_util.create_image_data_gen())

def make_gen_test():
    return icon_cate_util.datagenerator(aial_test, batch_size, epochs=1, cate_only=True, shuffle=False)


my_fit_generator(model, make_gen_train, make_gen_test, 10, batch_size = batch_size, callbacks = [])