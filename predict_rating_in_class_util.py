from icon_cate_util import create_icon_cate_model, filter_aial_rating_cate
from global_util import load_pickle
from keras_util import gen_k_fold_pass
import keras_util
import icon_cate_util
import icon_util
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
import math
import mypath
import matplotlib.pyplot as plt
import plt_util
import numpy as np
import sc_util
from predict_rating_in_regression_util import create_regression_rating_model_from_pretrained_model
from tensorflow.keras.callbacks import ModelCheckpoint

def _override_last_dense_layer(model, new_dense_size):
    input_layer = model.layers[0].input
    last_layer = model.layers[-2].output
    last_layer = Dense(new_dense_size, name='class_rating_dense', activation='softmax')(last_layer)
    return Model(inputs=[input_layer], outputs=[last_layer])

def compile_model_decorator(fn):
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        model.compile(optimizer='adam',
            loss='categorical_crossentropy', metrics=['acc'])
        return model
    return wrapper

@compile_model_decorator
def  create_class_rating_model(create_cate_model_args, class_n):
    model = create_icon_cate_model(**create_cate_model_args)
    return _override_last_dense_layer(model, class_n)

@compile_model_decorator
def create_class_rating_model_from_pretrained_model(pretrained_model, class_n):
    return _override_last_dense_layer(pretrained_model, class_n)
    
if __name__ == '__main__':
    split_period = [3.5 , 4, 4.5, 5]
    k_iter = 0
    batch_size=32
    epochs=1

    mypath.icon_folder = 'icons.combine.recrawled/'
    mypath.screenshot_folder = 'C:/screenshots.distincted.rem.human.zip/0/'

    aial = load_pickle('aial_seed_327.obj')
    aial = filter_aial_rating_cate(aial)

    pretrained_model = load_model('C:/Users/chaya/Downloads/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5')        
    model = create_regression_rating_model_from_pretrained_model(pretrained_model)
    model.load_weights('C:/Users/chaya/Downloads/icon_reg_rating_model2.4_k0_dj-ep-027-loss-0.096-mape-5.852-vloss-0.105-vmape-6.083.hdf5')
    # sc_dict = sc_util.make_sc_dict('C:/screenshots.distincted.rem.human.zip/0/')
    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k_iter, n_splits=4)
    # aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

    # f = open('t.txt', mode='w')
    # for _, rating, *_ in aial_test:
    #     print(rating, file=f)
    # f.close()
    # input('done')


    gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs,
        datagen = keras_util.create_image_data_gen(),
        predict_rating=True)
    gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs,
        shuffle=False,
        predict_rating=True)
    
    # xs, ys = [], []
    # for x,y in gen_test:
    #     for a in x: xs.append(a)
    #     for b in y: ys.append(b)
    
    
    f = open('t3.txt', mode='w')
    for app_id, rating, *_ in aial_test:
        img = icon_util.load_icon_by_fn('icons.combine.recrawled/' + app_id + '.png', 128, 128) / 255
        pred = model.predict(np.array([img]))[0][0]
        print(app_id, pred, file=f)
    f.close()
    # pred = model.predict(np.array(xs), 1)
    # f = open('t2.txt', mode='w')
    # for p in pred:
    #     print(p[0], file=f)
    # f.close()
    # input('done')


    # xs, ys = [], []
    # for x,y in gen_test:
        # for a in x: xs.append(a)
        # for b in y: ys.append(b)

    # plt_util.plot_confusion_matrix(model, (np.array(xs), np.array(ys)), 32)
    # model.evaluate(gen_test, steps = math.ceil(len(aial_test_sc)/batch_size))


    # model = create_class_rating_model_from_pretrained_model(pretrained_model, 4)
    # cw = icon_cate_util.compute_class_weight_for_class_rating(aial_train, split_period)
    # print(cw)
    # cw = icon_cate_util.compute_class_weight_for_class_rating(aial_test, split_period)
    # input()

    # checkpoint = ModelCheckpoint('model.hdf5', monitor='mape', save_best_only=True, verbose=0, period=1)

    # model.fit_generator(gen_train,
    # steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    # validation_data=gen_test,
    # validation_steps=math.ceil(len(aial_test)/batch_size),
    # epochs=epochs,
    # callbacks=[checkpoint])