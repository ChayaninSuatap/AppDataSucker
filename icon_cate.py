import preprocess_util
import icon_cate_util
import numpy as np
import random
import math
from keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass

random.seed(281)
np.random.seed(281)
aial = preprocess_util.prep_rating_category_scamount_download()
aial = preprocess_util.remove_low_rating_amount(aial, 100)
#filter only rating cate
newaial = []
for x in aial:
    newaial.append( (x[0], x[1], x[2]))
aial = newaial
random.shuffle(aial)

aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=0, n_splits=4)
print(icon_cate_util.compute_baseline(aial_train, aial_test))

model = icon_cate_util.create_icon_cate_model(cate_only=True)

batch_size = 24
epochs = 999
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True)

filepath='reg_cate_only_k0-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
### save for predict rating + cate
# filepath='reg_cate_only_k0-ep-{epoch:03d}-loss-{my_model_regress_1_loss:.3f}-vloss-{val_my_model_regress_1_loss:.3f}-vmape-{val_my_model_regress_1_mean_absolute_percentage_error:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)
