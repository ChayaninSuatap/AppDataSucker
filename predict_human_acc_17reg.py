from tensorflow.keras.models import load_model, Model
from predict_human_acc import add_dense
from tensorflow.keras.layers import Dense
import global_util
import icon_util
import keras_util
import numpy as np
from predict_human_acc_binary import best_epoch_by_val

def mod_model(model, freeze, denses):
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

    output_layer = Dense(17, activation='softmax')(x)

    model = Model(input_layer.input, output_layer)
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model

def make_human_acc_from_lines(output_path, is_sc):
    output = {}
    for i in range(340):
        line = input()
        cells = line.split('\t')
        sum_cates = np.zeros(17)
        for j, cell in enumerate(cells):
            if j % 2 == 1: continue
            if not is_sc and j == 16: continue
            sum_cates[int(cell)] += 1
        output[str(i) + '.png'] = sum_cates/9
    print(output)
    global_util.save_pickle(output, output_path)

def make_icon_human_acc_obj_from_lines(output_path = 'journal/pred_human_acc/icon_human_acc_17reg.obj'):
    print('insert icon cells:')
    make_human_acc_from_lines(output_path, is_sc=False)

def make_sc_human_acc_obj_from_lines(output_path = 'journal/pred_human_acc/sc_human_acc_17reg.obj'):
    print('insert sc cellss:')
    make_human_acc_from_lines(output_path, is_sc=True)

def make_icon_dataset(icon_human_acc_obj, icon_dir = 'icons_human_test/'):
    output = []
    for icon_fn, label in icon_human_acc_obj.items():
        icon = icon_util.load_icon_by_fn(icon_dir + icon_fn, 128, 128) / 256
        output.append( (icon, label))
    return output

def make_sc_dataset(sc_human_acc_obj, sc_dir = 'screenshots_human_test/'):
    output = []
    for sc_fn, label in sc_human_acc_obj.items():
        sc = icon_util.load_icon_by_fn(sc_dir + sc_fn, 256, 160, rotate_for_sc=True) / 256
        output.append( (sc, label))
    return output

if __name__ == '__main__':
    d = global_util.load_pickle('journal/pred_human_acc/icon_human_acc_17reg.obj')
    dat = make_icon_dataset(d)
    # global_util.save_pickle(dat, 'journal/pred_human_acc/sc_feature_human_acc_17reg.obj')
    epochs = 50
    batch_size = 16
    freeze = True
    denses = []
    model = load_model('sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5')
    model = mod_model(model, freeze, denses)

    for k_iter in range(4):
        x_train, y_train, x_test, y_test = keras_util.gen_k_fold_pass_as_np(aial=dat, kf_pass=k_iter, n_splits=4)
        history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test),
        epochs=epochs, batch_size=batch_size, verbose=1)
        input(best_epoch_by_val(history, 'val_mae', ['val_loss', 'val_mae']))
