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
import os

def load_icon_human_test_set():
    o = global_util.load_pickle('app_ids_for_human_test.obj')
    xs = []
    ys = []
    for app_id, class_num in o:
        print(app_id, class_num)
        icon = icon_util.load_icon_by_fn('icons.backup/'+app_id+'.png', 128, 128)
        icon = icon.astype('float32')
        icon/=255
        xs.append(icon)
        y = [0] * 17
        y[class_num] = 1
        ys.append(y)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

batch_size = 32

mypath.icon_folder = 'similarity_search/icons_rem_dup_human_recrawl/'
mypath.screenshot_folder = 'screenshots.256.distincted.rem.human/'

#load human icon dataset
# hxs, hys = load_icon_human_test_set()

sc_dict = sc_util.make_sc_dict('screenshots.256.distincted.rem.human/')

for dir in os.listdir('ensemble_models_t'):
    for model_fn in os.listdir('ensemble_models_t/'+dir):

        #FIXME pred screenshot
        if dir[:3] != 'sc_':continue

        model_fd = 'ensemble_models_t/'+dir+'/'
        model_path = model_fd + model_fn

        model_save_path = 'ensemble_model_predicts_t/' + model_fn.split('-')[0] + '.obj'
        #human test set
        # model_save_path = 'ensemble_model_predicts_t/' + model_fn.split('-')[0] + '_human.obj'

        model = load_model(model_path)

        #predict normal
        random.seed(327)
        np.random.seed(327)

        splited_0 = model_fn.split('-')[0]
        if '_k0_' in splited_0: k = 0
        if '_k1_' in splited_0: k = 1
        if '_k2_' in splited_0: k = 2
        if '_k3_' in splited_0: k = 3
        aial = icon_cate_util.make_aial_from_seed(327, mypath.icon_folder)
        aial = icon_cate_util.filter_aial_rating_cate(aial)
        aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k, n_splits=4)

        # gen_test = icon_cate_util.datagenerator(aial_test, batch_size, 1, cate_only=True, shuffle=False, enable_cache=True)

        aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)
        gen_test=icon_cate_util.datagenerator(aial_test_sc,
            batch_size, 1, cate_only=True, train_sc=True, shuffle=False)

        print('predicting', model_fn, 'k=', k)
        preds = model.predict_generator(gen_test, steps = math.ceil(len(aial_test_sc) / batch_size))
        
        #compute acc
        acc = 0
        for pred_argmax, x in zip(preds.argmax(axis = 1), aial_test_sc):
            cate = x[2]
            if cate[pred_argmax] == 1:
                acc += 1
        print('acc',acc/len(aial_test_sc))

        print('complete', 'len preds', len(preds), 'len aial_test', len(aial_test_sc))
        global_util.save_pickle(preds, model_save_path)

        #eval for human test
        # if '_k0_' not in splited_0: continue
        # print('start pred', model_fn)
        # print(model.evaluate(hxs, hys))
        # preds = model.predict(hxs)
        # global_util.save_pickle(preds, model_save_path)

