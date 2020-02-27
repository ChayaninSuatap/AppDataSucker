import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass, metric_top_k, eval_top_5, compute_class_weight
from tensorflow.keras.models import load_model
import keras
import functools
import icon_cate_data_export
import global_util
import mypath
import keras_util
import matplotlib.pyplot as plt
import sc_util

k = 0
batch_size = 128
epochs = 100

random.seed(327)
np.random.seed(327)
mypath.icon_folder = 'similarity_search/icons_rem_dup_recrawl/'
aial = icon_cate_util.make_aial_from_seed(327, mypath.icon_folder)
aial = icon_cate_util.filter_aial_rating_cate(aial)
aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k, n_splits=4)

#icon
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=True, datagen=keras_util.create_image_data_gen())
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False, enable_cache=True)

#sc
# mypath.screenshot_folder = 'screenshots.256.distincted.rem.human/'
# sc_dict = sc_util.make_sc_dict()
# aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

# gen_train=icon_cate_util.datagenerator(aial_train_sc,
#         batch_size, epochs, cate_only=True, train_sc=True, enable_cache=False, limit_cache_n=45000, datagen=keras_util.create_image_data_gen())
# gen_test=icon_cate_util.datagenerator(aial_test_sc,
#         batch_size, epochs, cate_only=True, train_sc=True, shuffle=False)

# model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False,
#                                               layers_filters = [64, 128, 256], dropout = 0.2)
model = load_model('C:/Users/chaya/Downloads/icon_model1_k0_t-ep-266-loss-0.507-acc-0.830-vloss-3.467-vacc-0.319.hdf5')
eval_top_5(model, gen_test, math.ceil(len(aial_test)/batch_size))

# model.fit_generator(gen_train,
#     steps_per_epoch=math.ceil(len(aial_train_sc)/batch_size),
#     validation_data=gen_test,
#     validation_steps=math.ceil(len(aial_test_sc)/batch_size),
#     epochs=epochs)


#old stuff
# aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
# aial = preprocess_util.remove_low_rating_amount(aial, 100)
# random.shuffle(aial)
# print('aial loss',icon_cate_util.compute_aial_loss(aial))
# icon_cate_util.check_aial_error(aial)

# #filter only rating cate
# aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)

# aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=3, n_splits=4)

# print(icon_cate_util.compute_baseline(aial_train, aial_test))
# class_weight = icon_cate_util.compute_class_weight_for_cate(aial_test)
# print(class_weight)

# model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, layers_filters = [64, 128, 256], conv
    # conv1x1_reduce_rate=2, sliding_dropout=(0.05,0.05))
# model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False,
#     layers_filters = [64, 128, 256, 512], layers_dense=[128, 68, 34], disable_conv_dropout=True, sliding_dropout=[0.1, 0.2])
# input()
# model = load_model('cate_model5_k3-ep-429-loss-0.026-acc-0.992-vloss-5.426-vacc-0.362.hdf5')

#export
# icon_cate_data_export.predict_for_spreadsheet(model, 0, aial_test)
# icon_cate_data_export.predict_combine_v1(model, 0, aial_test, '1024')
# input()

#eval for human test
# o = global_util.load_pickle('app_ids_for_human_test.obj')
# xs = []
# ys = []
# for app_id, class_num in o:
#     print(app_id, class_num)
#     icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
#     icon = icon.astype('float32')
#     icon/=255
#     xs.append(icon)
#     y = [0] * 17
#     y[class_num] = 1
#     ys.append(y)
# xs = np.array(xs)
# ys = np.array(ys)
# print(xs.shape)
# print(model.evaluate(xs, ys))
# print('start pred')
# pred = model.predict(xs).argmax(axis=1)
# for x in pred:
#     print(x)
# input()

# batch_size = 16
# epochs = 999
# gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=True, predict_rating=True)
# gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False, enable_cache=True, predict_rating=True)

# # test generator stability
# # arr = [[]]
# # for i in range(999):
# #     arr.append([])
# #     for j in range(math.ceil(len(aial_test)/batch_size)):
# #         xs = gen_test.__next__()
# #         arr[i] += [x.argmax() for x in xs[1]]
# #     print(hash(frozenset(arr[i])))
# # for i in range(len(arr[0])):
# #     if arr[0][i] != arr[2][i]: print('fail')
# # input('its fine maybe')

# filepath='t-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_mean_absolute_percentage_error:.3f}.hdf5'

# # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
# palc = PlotAccLossCallback(is_cate=False, is_regression=True)
# model.fit_generator(gen_train,
#     steps_per_epoch=math.ceil(len(aial_train)/batch_size),
#     validation_data=gen_test, max_queue_size=1,
#     validation_steps=math.ceil(len(aial_test)/batch_size),
#     callbacks=[palc], epochs=epochs, initial_epoch=0,)
