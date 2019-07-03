import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass, metric_top_k, eval_top_5
from keras.models import load_model
import keras
import functools
import icon_cate_data_export
import global_util

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

model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, layers_filters = [64, 128, 256, 512, 1024],
    sliding_dropout=(0.05,0.05), conv1x1_layer_n=2)
model.load_weights('cate_conv_1024_gap_k0-ep-480-loss-0.024-acc-0.993-vloss-5.465-vacc-0.372.hdf5')

#export
# icon_cate_data_export.predict_for_spreadsheet(model, 0, aial_test)
# input()

#eval for human test
o = global_util.load_pickle('app_ids_for_human_test.obj')
xs = []
ys = []
for app_id, class_num in o:
    print(app_id, class_num)
    icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
    icon = icon.astype('float32')
    icon/=255
    xs.append(icon)
    y = [0] * 17
    y[class_num] = 1
    ys.append(y)
xs = np.array(xs)
ys = np.array(ys)
print(xs.shape)
print(model.evaluate(xs, ys))
print('start pred')
pred = model.predict(xs).argmax(axis=1)
for x in pred:
    print(x)
input()

batch_size = 16
epochs = 999
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=True)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False, enable_cache=True)

#eval top 5 acc
eval_top_5(model, gen_test, math.ceil(len(aial_test)/batch_size))
input()


filepath='t-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
### save for predict rating + cate
# filepath='reg_cate_only_k0-ep-{epoch:03d}-loss-{my_model_regress_1_loss:.3f}-vloss-{val_my_model_regress_1_loss:.3f}-vmape-{val_my_model_regress_1_mean_absolute_percentage_error:.3f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs, initial_epoch=0)
