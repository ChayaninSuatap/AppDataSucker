import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass
from keras.models import load_model
import keras
import functools

random.seed(859)
np.random.seed(859)
aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
aial = preprocess_util.remove_low_rating_amount(aial, 100)
random.shuffle(aial)
print('aial loss',icon_cate_util.compute_aial_loss(aial))
fds = icon_cate_util._make_fds(aial)
for x in fds: x.show()
for x in aial:
    if all(y==0 for y in x[2]): print('all zero')
    if sum(x[2])>1: print('shit')
input('done')

#filter only rating cate
newaial = []
for x in aial:
    newaial.append( (x[0], x[1], x[2]))
aial = newaial

aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=0, n_splits=4)
print(icon_cate_util.compute_baseline(aial_train, aial_test))

model = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, use_gap=True)

batch_size = 24
epochs = 999
gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True)
gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True)

#predict
#last ep 6
# model = load_model('cate_only_softmax-ep-211-loss-0.067-acc-0.979-vloss-5.002-vacc-0.292.hdf5')
# f = open('pred_ep_20.txt', 'w')
# f2 = open('real_ep_20.txt', 'w')
# for x in aial_test:
#     app_id = x[0]
#     answer = x[2]
#     try:
#         icon = icon_util.load_icon_by_app_id(x[0], 128, 128)
#     except Exception as e:
#         pass
#     finally:
#         icons = np.array([icon])
#         icons = icons.astype(np.float)
#         icons /= 255  
#         pred = model.predict(icons)
#         f.writelines(str(pred[0].tolist()) + '\n')
#         f2.writelines(str(answer) + '\n')
# input('done')

#eval top k
model = load_model('cate_only_softmax-ep-050-loss-0.204-acc-0.935-vloss-3.892-vacc-0.287.hdf5')
top_3 = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top_5 = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top_3.__name__ = 'top_3'
top_5.__name__ = 'top_5'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', top_3, top_5])
print(model.evaluate_generator(gen_test,steps=math.ceil(len(aial_test)/batch_size)))
input()

filepath='reg_cate_only_softmax_gap_k0-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
### save for predict rating + cate
# filepath='reg_cate_only_k0-ep-{epoch:03d}-loss-{my_model_regress_1_loss:.3f}-vloss-{val_my_model_regress_1_loss:.3f}-vmape-{val_my_model_regress_1_mean_absolute_percentage_error:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)
