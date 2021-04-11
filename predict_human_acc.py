from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Conv2D, Input, LeakyReLU, BatchNormalization, Dropout
import global_util
import os
import icon_util
import numpy as np
import keras_util
from tensorflow.keras.callbacks import ModelCheckpoint

def add_dense(size, last_layer, dropout_rate = 0.5, level=0):
    x = Dense(size)(last_layer)
    x = LeakyReLU(name=('pred_human_acc_LeakyReLu' + str(level)))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    return x

def mod_model(model, freeze, denses=[64]):
    #get flaten layer
    input_layer = None
    flatten_layer = None

    for layer in model.layers:
        if layer.name == 'input_1':
            input_layer = layer
        elif layer.name == 'my_model_flatten':
            flatten_layer = layer
            break
            
        #freeze
        if freeze:
            layer.trainable = False
    
    x = flatten_layer.output
    for level, dense in enumerate(denses):
        x = add_dense(dense, x, level=level)

    output_layer = Dense(1, activation='sigmoid')(x)

    model = Model(input_layer.input, output_layer)
    model.compile(loss='mse', optimizer='adam')
    return model

def make_human_acc_obj_from_lines(output_path = 'journal/pred_human_acc/icon_human_acc.obj'):
    output = {}
    for i in range(340):
        line = input()
        app_id, acc = line.split('\t')[0], float(line.split('\t')[1])
        output[app_id] = acc
    global_util.save_pickle(output, output_path)

def make_sc_human_acc_obj_from_lines(output_path = 'journal/pred_human_acc/sc_human_acc.obj'):
    output = {}
    for i in range(340):
        line = input()
        acc = float(line)
        output[str(i) + '.png'] = acc
    print(output)
    global_util.save_pickle(output, output_path)

def make_icon_dataset(icon_human_acc_obj, icon_dir = 'icons.combine.recrawled/'):
    output = []
    for app_id, avg_human_acc in icon_human_acc_obj.items():
        icon = icon_util.load_icon_by_fn(icon_dir + app_id + '.png', 128, 128) / 256
        output.append( (icon, avg_human_acc))
    return output

def make_sc_dataset(sc_human_acc_obj, sc_dir = 'screenshots_human_test/'):
    output = []
    for sc_fn, avg_human_acc in sc_human_acc_obj.items():
        sc = icon_util.load_icon_by_fn(sc_dir + sc_fn, 256, 160, rotate_for_sc=True) / 256
        output.append( (sc, avg_human_acc))
    return output

def best_val_loss(history):
    idx = None
    minv = None
    for i,x in enumerate(history.history['val_loss']):
        if minv is None or x < minv:
            minv = x
            idx = i

    return history.history['val_loss'][idx], idx

if __name__ == '__main__':
    # epochs = 100
    # batch_size = 16
    # denses = [32]
    # project = 'icon_sigmoid_32_notfreeze'
    # freeze = False

    # for k_iter in range(4):
    #     model = load_model('sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5')
    #     model = mod_model(model, freeze=freeze, denses=denses)

    #     app_id_avg_acc_d = global_util.load_pickle('journal/pred_human_acc/icon_human_acc.obj')
    #     dat = make_icon_dataset(app_id_avg_acc_d)
    #     x_train, y_train, x_test, y_test = keras_util.gen_k_fold_pass_as_np(aial=dat, kf_pass=k_iter, n_splits=4)

    #     checkpoint_save_path = 'journal/pred_human_acc/%s_k%d.hdf5' % (project, k_iter,)
    #     checkpoint = ModelCheckpoint(checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)
    #     history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
    #     epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=0)
    #     print(best_val_loss(history))

    # predict test set
    # model = load_model('journal/pred_human_acc/icon_sigmoid_16_k0.hdf5')
    # app_id_avg_acc_d = global_util.load_pickle('journal/pred_human_acc/icon_human_acc.obj')
    # dat = make_icon_dataset(app_id_avg_acc_d)
    # x_train, y_train, x_test, y_test = keras_util.gen_k_fold_pass_as_np(aial=dat, kf_pass=0, n_splits=4)

    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # pred_result = model.predict(x_test, batch_size = 1)
    # for x in pred_result: print(x[0])
    # eval_result = model.evaluate(x_test, y_test, batch_size=1)
    # print(eval_result)

    # for x in y_test: print(x)

    


    epochs = 100
    batch_size = 8
    denses = [16]
    project = 'sc_sigmoid_16_nofreeze'
    freeze = False

    for k_iter in range(4):
        model = load_model('sim_search_t/models/sc_model2.3_k3_no_aug-ep-085-loss-0.786-acc-0.761-vloss-2.568-vacc-0.403.hdf5')
        model = mod_model(model, freeze=freeze, denses=denses)

        app_id_avg_acc_d = global_util.load_pickle('journal/pred_human_acc/sc_human_acc.obj')
        dat = make_sc_dataset(app_id_avg_acc_d)
        x_train, y_train, x_test, y_test = keras_util.gen_k_fold_pass_as_np(aial=dat, kf_pass=k_iter, n_splits=4)

        checkpoint_save_path = 'journal/pred_human_acc/%s_k%d.hdf5' % (project, k_iter,)
        checkpoint = ModelCheckpoint(checkpoint_save_path, monitor='val_loss', save_best_only=True, verbose=0)
        history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
        epochs=epochs, batch_size=batch_size, callbacks=[checkpoint], verbose=0)
        print(best_val_loss(history))
