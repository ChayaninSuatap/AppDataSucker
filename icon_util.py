import numpy as np
from PIL import Image
import mypath
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def load_icon_by_app_id(app_id, resizeW, resizeH):
    return open_and_resize(mypath.icon_folder + app_id + '.png', resizeW, resizeH)

def load_icon_by_fn(fn, resizeW, resizeH, rotate_for_sc=False):
    return open_and_resize(fn, resizeW, resizeH, rotate_for_sc)

def open_and_resize(fn, resizeW, resizeH, rotate_for_sc=False):
    converted = _convert_to_rgba(fn, resizeW, resizeH, rotate_for_sc)
    return np.array(converted)[:,:,:3]

def _convert_to_rgba(fn, resizeW, resizeH, rotate_for_sc=False):
    #train screenshot
    if rotate_for_sc:
        png = Image.open(fn).convert('RGB')
        w,h = png.size
        #rotate
        if h > w:
            png = png.rotate(90, expand=True)
        #resize only when needed
        if png.size == (resizeW, resizeH):
            return png
        else:
            return png.resize( (resizeW, resizeH))
    #train icon
    else:
        png = Image.open(fn).convert('RGBA')
        png = png.resize( (resizeW, resizeH))
        background = Image.new('RGBA', png.size, (255,255,255))
        alpha_composite = Image.alpha_composite(background, png)
        return alpha_composite

def oversample_image(app_ids_and_labels):
    app_id_pool = {}
    label_counter = {}
    for app_id, label in app_ids_and_labels:
        #add in pool
        if label not in app_id_pool: app_id_pool[label] = []
        app_id_pool[label].append((app_id,label))
        if label  not in label_counter: label_counter[label] = 0
        label_counter[label] += 1
    #start sampling
    max_freq = max( list(label_counter.values()))
    for label in label_counter.keys():
        for i in range(max_freq - label_counter[label]):
            #pick and app_id
            picked = random.choice(app_id_pool[label])
            app_ids_and_labels.append( picked)

def create_model(IS_REGRESSION, summary=False, use_gap=False, train_sc=False, layers_filters = [64, 128, 256], dropout=0.2,
    conv1x1_no_maxpool=False):
    def add_conv(layer, filter_n, kernel_size=(3,3), dropout=0.2, padding_same=False, maxpool=True):
        padding = 'same' if padding_same else 'valid'
        x = Conv2D(filter_n ,kernel_size, padding=padding)(layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        if maxpool:
            x = MaxPooling2D()(x) 
        return x
    #make model
    #define input layer
    if train_sc:
        input_layer = Input(shape=(160, 256, 3))
    else:
        input_layer = Input(shape=(128, 128, 3))
    #define conv layers
    x = add_conv(input_layer, layers_filters[0], padding_same=True, dropout=dropout)
    for i in range(1, len(layers_filters)):
        x = add_conv(x, layers_filters[i], padding_same=True, dropout=dropout)
    #connect with GAP or a conv1x1 layer
    if not use_gap:
        if conv1x1_no_maxpool == False:
            x = add_conv(x, layers_filters[-1]//2, padding_same=True, kernel_size=(1,1), dropout=dropout)
        elif conv1x1_no_maxpool == True:
            x = add_conv(x, layers_filters[-1]//2, padding_same=True, kernel_size=(1,1), dropout=dropout, maxpool=False)
        x = Flatten(name='my_model_flatten')(x)
        flatten_layer = x
    if use_gap:
        x = GlobalAveragePooling2D()(x)
        # when use_gap link to gap instead
        flatten_layer = x

    #output layer
    if IS_REGRESSION:
        x = Dense(16, name='my_model_dense_1')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        x = Dense(4, name='my_model_dense_2')(x)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        output_layer = Dense(1, activation='linear', name='my_model_regress_1')(x)
    else:
        x = Dense(16, name='my_model_dense_2')(x)
        x = LeakyReLU()(x)
        output_layer = Dense(3, activation='softmax', name='my_model_dense_3')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    #compile
    if IS_REGRESSION:
        model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    if summary : model.summary()

    return {'model':model, 'flatten_layer':flatten_layer, 'input_layer':input_layer, 'output_layer':output_layer}

