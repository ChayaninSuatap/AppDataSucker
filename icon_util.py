import numpy as np
from PIL import Image
import mypath
import random
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU
from keras.models import Model

def load_icon_by_app_id(app_id, resizeW, resizeH):
    return open_and_resize(mypath.icon_folder + app_id + '.png', resizeW, resizeH)

def open_and_resize(fn, resizeW, resizeH):
    return np.asarray( _convert_to_rgba(fn, resizeW, resizeH ))[:,:,:3]

def _convert_to_rgba(fn, resizeW, resizeH):
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

def create_model(IS_REGRESSION):
    def add_conv(layer, filter_n, kernel_size=(3,3), dropout=0.2):
        x = Conv2D(filter_n ,kernel_size)(layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D()(x) 
        return x
    #make model
    input_layer = Input(shape=(128, 128, 3))
    x = add_conv(input_layer, 64)
    x = add_conv(x, 128)
    x = add_conv(x, 256)
    x = add_conv(x, 128, kernel_size=(1,1))
    x = Flatten(name='my_model_flatten')(x)

    #output layer
    if IS_REGRESSION:
        x = Dense(16, name='my_model_dense_1')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = Dense(4, name='my_model_dense_2')(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = Dense(1, activation='linear', name='my_model_regress_1')(x)
    else:
        x = Dense(16, name='my_model_dense_2')(x)
        x = LeakyReLU()(x)
        x = Dense(3, activation='softmax', name='my_model_dense_3')(x)
    model = Model(input=input_layer, output=x)

    #compile
    if IS_REGRESSION:
        model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_percentage_error'])
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model