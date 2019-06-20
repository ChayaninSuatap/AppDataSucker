import os
import mypath
import random
import numpy as np
import preprocess_util
import icon_cate_util
from keras_util import gen_k_fold_pass
import sc_util
from tensorflow.keras.callbacks import ModelCheckpoint
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

for i in range(4):
    sum_train = [0] * 17
    sum_test = [0] * 17
    aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=i, n_splits=4)
    for x in aial_train:
        sum_train[x[2].index(1)] += 1
    for x in aial_test:
        sum_test[x[2].index(1)] += 1
    print(sum_train)
    print(sum_test)
    input()
print(icon_cate_util.compute_baseline(aial_train, aial_test))

#make aial_train_sc, aial_test_sc
aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

batch_size = 8
epochs = 999
gen_train=icon_cate_util.datagenerator(aial_train_sc,
        batch_size, epochs, cate_only=True, train_sc=True)
gen_test=icon_cate_util.datagenerator(aial_test_sc,
        batch_size, epochs, cate_only=True, train_sc=True)

elem = max( [(k,v) for k,v in sc_dict.items()], key=lambda x: len(x[1]))
print(elem)
input()
from keras.models import load_model
import icon_util
model = load_model('sc_cate_only_softmax_k0-ep-061-loss-0.249-acc-0.916-vloss-3.464-vacc-0.359.hdf5')
for app_id, _ , label in aial_test:
    if (app_id in sc_dict) == False: continue
    truth = np.array(label).argmax()
    #output
    print(app_id, truth, end=' ')
    file_out = app_id + ' ' + str(truth) + ' '

    for ss_fn in sc_dict[app_id]:
        try:
            icon = icon_util.load_icon_by_fn(mypath.screenshot_folder+ss_fn, 256, 160, rotate_for_sc=True)
        except:
            continue
        icon = icon.astype('float32')
        icon /= 255
        pred = model.predict(np.array([icon]))
        t = pred[0].argmax()
        #output
        print(t, end=' ')
        file_out += str(t) + ' '
    #output
    print('')
    f = open('sc_fold1_testset.txt', 'a')
    f.writelines(file_out + '\n')
    f.close()

input('done')

            



# model=icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=True)
filepath='sc_cate_only_softmax_k0-ep-{epoch:03d}-loss-{loss:.3f}-acc{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback(is_cate=False)
model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train_sc)/batch_size),
    validation_data=gen_test, max_queue_size=1,
    validation_steps=math.ceil(len(aial_test_sc)/batch_size),
    callbacks=[checkpoint, palc], epochs=epochs)

