import os
import mypath
import random
import numpy as np
import preprocess_util
import icon_cate_util
from keras_util import gen_k_fold_pass
import sc_util
from keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback
import math

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
        batch_size, epochs, cate_only=True, train_sc=True)

model=icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=True)
filepath='sc_cate_only_softmax_k0-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train_sc)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test_sc)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)

