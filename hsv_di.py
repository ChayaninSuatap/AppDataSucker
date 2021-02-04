from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from icon_util import open_and_resize
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Dropout, LeakyReLU, concatenate
from download_increase_util import prepare_dataset, compute_class_weight, make_confmat
import random
import keras_util
import global_util
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import confusion_matrix
from overall_feature_download_increase_util import best_val_acc

def make_feature(fn, bin=10, img_size=None):
    img = Image.open(fn)

    if img_size is not None:
        width, height = img_size
    else:
        width ,height = img.size

    img = open_and_resize(fn, width, height)
    img = np.array(img).astype('float') / 255
    img_hsv = rgb_to_hsv(img)

    hue = np.zeros(bin)
    sat = np.zeros(bin)
    val = np.zeros(bin)
    split = 1.0/bin

    for row in img_hsv:
        for col in row:
            h,s,v = col
            hue[int(h // split)] += 1
            sat[int(s // split)] += 1
            val[int(v // split)] += 1
    hue_norm = (hue / (width * height))
    sat_norm = (sat / (width * height))
    val_norm = (val / (width * height))
    feature = np.concatenate((hue_norm,sat_norm,val_norm))
    return feature

def make_model(input_size, denses=[8,4]):

    def add_dense(input_layer, dense_size, dropout_rate=0.5):
        x = Dense(dense_size)(input_layer)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        return x

    input_layer = Input(shape=(input_size,))
    x = input_layer
    for dense in denses:
        x = add_dense(x, dense)
    x = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    return model

def make_icon_dataset(
    bin_split,
    output_path,
    icon_path = 'icons_rem_dup_human_recrawl/icons_rem_dup_recrawl/',
):
    dat = prepare_dataset()
    random.seed(5)
    random.shuffle(dat)

    processed_dat = []

    for app_id, label in dat:
        print(app_id)
        feature = make_feature(icon_path + app_id + '.png', bin_split, (180, 180))
        processed_dat.append( (feature, label))
    global_util.save_pickle(processed_dat, output_path)
    print('finished creating features')

def undersample(train_data):
    samples0 = []
    samples1 = []
    new_train_data = [[],[],[]]
    for x in train_data:
        if x[1][1] == 0:
            samples0.append(x)
        else:
            samples1.append(x)
    
    random.shuffle(samples0)
    for i in range(len(samples0)):
        modded = i % 3
        new_train_data[modded].append(samples0[i])
    
    for x in samples1:
        new_train_data[0].append(x)
        new_train_data[1].append(x)
        new_train_data[2].append(x)

    random.shuffle(new_train_data[0])
    random.shuffle(new_train_data[1])
    random.shuffle(new_train_data[2])

    return new_train_data

if __name__ == '__main__':
    # f = make_feature('hsv_di/test hsv/yellow.png', 20)
    # print(f)
    # input('done')

    icon_path = 'icons_rem_dup_human_recrawl/icons_rem_dup_recrawl/'
    bin_split = 5
    output_path = 'hsv_di/features/bin5.obj'
    # make_icon_dataset(bin_split, output_path=output_path)
    # input('done')

    save_folder = 'hsv_di/models/'
    model_name = 'bin5'
    batch_size = 32
    epochs = 50
    model_input_size = bin_split * 3
    denses = [ 15, 8, 4]
    is_undersample = True

    # make_icon_dataset(bin_split, output_path)

    processed_dat = global_util.load_pickle(output_path)
    confmats = []
    best_accs = []
    best_eps = []

    for k_iter in range(0):
        (train_data , test_data) = keras_util.gen_k_fold_pass(processed_dat, kf_pass = k_iter, n_splits=4)

        train_data = undersample(train_data)
        #global.save ....

        # c0, c1 = 0, 0
        # for _,x in train_data:
        #     if x[0] == 1: c0+=1
        #     else: c1+=1
        # print(c0,c1)
        # input()

        train_features = []
        train_labels = []
        for feature,label in train_data:
            train_features.append(feature)
            train_labels.append(label)
        
        test_features = []
        test_labels = []
        for feature,label in test_data:
            test_features.append(feature)
            test_labels.append(label)
        
        #normalize
        train_features, test_features, _ = keras_util.standard_normalize(
            train_features,
            test_features
        )

        train_features = np.array(train_features)
        test_features = np.array(test_features)
        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)

        cw = compute_class_weight(train_labels)

        model = make_model(model_input_size, denses=denses)
        cp_best_ep = ModelCheckpoint('%s%s_k%d.hdf5' % (save_folder, model_name, k_iter), monitor='val_acc', save_best_only=True, verbose=0)
        history = model.fit(x=train_features, y=train_labels,
            validation_data=(test_features, test_labels),
            batch_size=batch_size, epochs=epochs, callbacks=[cp_best_ep], class_weight=cw, verbose=0)

        top_val_acc, top_val_ep = best_val_acc(history)
        print('%.3f %d' % (top_val_acc, top_val_ep))
        best_accs.append(top_val_acc)
        best_eps.append(top_val_ep)

        #show confmat
        model = load_model('%s%s_k%d.hdf5' % (save_folder, model_name, k_iter))
        preds = model.predict(test_features)

        confmat = make_confmat(test_labels, preds)
        # print('confmat', k_iter)
        # print(confmat)
        confmats.append(confmat)
    
    print('final result')
    for acc, ep in zip(best_accs, best_eps):
        print('%.3f %d' % (acc, ep))
    final_confmats = sum(confmats)/4
    print(final_confmats[0][0], final_confmats[0][1])
    print(final_confmats[1][0], final_confmats[1][1])

    prec = final_confmats[1][1]/(final_confmats[0][1] + final_confmats[1][1])
    recall = final_confmats[1][1]/(final_confmats[1][1] + final_confmats[1][0])
    f1 = (prec*recall*2)/(prec+recall)
    print('%.3f' % (f1,))


