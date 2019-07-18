import os
import mypath
import random
import numpy as np
import preprocess_util
import icon_cate_util
from keras_util import gen_k_fold_pass
import sc_util
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras_util import PlotAccLossCallback
import math
import sc_data_export

sc_dict = sc_util.make_sc_dict()

random.seed(859)
np.random.seed(859)
aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
aial = preprocess_util.remove_low_rating_amount(aial, 100)
random.shuffle(aial)
print('aial loss',icon_cate_util.compute_aial_loss(aial))
icon_cate_util.check_aial_error(aial)

#filter only rating cate
aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=0, n_splits=4)

print(icon_cate_util.compute_baseline(aial_train, aial_test))

#make aial_train_sc, aial_test_sc
aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

batch_size = 8
epochs = 999
gen_train=icon_cate_util.datagenerator(aial_train_sc,
        batch_size, epochs, cate_only=True, train_sc=True)
gen_test=icon_cate_util.datagenerator(aial_test_sc,
        batch_size, epochs, cate_only=True, train_sc=True, shuffle=False)

# model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=True, layers_filters=[64,128,256,512])
# model.load_weights('sc_cate_conv_512_k0-ep-132-loss-0.097-acc-0.969-vloss-3.760-vacc-0.386.hdf5')
model = load_model('sc_cate_conv_1024_2_conv1x1_slide_do_k0-ep-274-loss-0.028-acc-0.991-vloss-4.660-vacc-0.417.hdf5')

#eval for human test
import global_util
import icon_util
o = global_util.load_pickle('app_ids_for_human_test.obj')
labels = [x for _,x in o]
xs = []
ys = []
for i, class_num in enumerate(labels):
    icon = icon_util.load_icon_by_fn('screenshots_human_test/' + str(i) + '.png', 256, 160, rotate_for_sc=True)
    icon = icon.astype('float32')
    icon/=255
    xs.append(icon)
    y = [0] * 17
    y[class_num] = 1
    ys.append(y)
xs = np.array(xs)
ys = np.array(ys)
print(xs.shape)
# print(model.evaluate(xs, ys))
print('start pred')
pred = model.predict(xs, batch_size=1).argmax(axis=1)
for x in pred:
    print(x)
input()

#export data
# sc_data_export.predict_for_spreadsheet(model ,
#     k_iter=0, aial_test = aial_test, sc_dict= sc_dict, fn_postfix='1024')
# input('done')

# model=icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=True)
filepath='sc_cate_only_softmax_k0-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train_sc)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test_sc)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)

