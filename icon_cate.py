import preprocess_util
import icon_cate_util
import numpy as np
import random
import math
from keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass

aial = preprocess_util.prep_rating_category()
random.seed(7)
np.random.seed(7)
random.shuffle(aial)
ninety = int(len(aial)*80/100)
aial_train = aial[:ninety]
aial_test = aial[ninety:]
aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=0, n_splits=10)
print(icon_cate_util.compute_baseline(aial_test))

model = icon_cate_util.create_icon_cate_model()

batch_size = 24
epochs = 999
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs)

filepath='reg_cate_padding_same_k1-ep-{epoch:03d}-loss-{my_model_regress_1_loss:.3f}-vloss-{val_my_model_regress_1_loss:.3f}-vmape-{val_my_model_regress_1_mean_absolute_percentage_error:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=True)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)
