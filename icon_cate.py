import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass, metric_top_k
from keras.models import load_model
import keras
import functools
import icon_cate_data_export

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

model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True)
print('worked')

#export
icon_cate_data_export.predict_for_spreadsheet('cate_only_softmax-ep-100-loss-0.113-acc-0.962-vloss-4.678-vacc-0.317.hdf5'
, 0, aial_test, model)
input()

batch_size = 16
epochs = 999
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False)

#eval top k
icon_cate_util.eval_top_k(gen_test, math.ceil(len(aial_test)/batch_size), model=model)
input()

model.load_weights('reg_cate_17_softmax_k3-ep-023-loss-0.548-acc0.818-vloss-3.618-vacc-0.284.hdf5')
# model = load_model('reg_cate_17_softmax_k3-ep-023-loss-0.548-acc0.818-vloss-3.618-vacc-0.284.hdf5')
filepath='reg_cate_17_softmax_k3-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
### save for predict rating + cate
# filepath='reg_cate_only_k0-ep-{epoch:03d}-loss-{my_model_regress_1_loss:.3f}-vloss-{val_my_model_regress_1_loss:.3f}-vmape-{val_my_model_regress_1_mean_absolute_percentage_error:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs, initial_epoch=23)
