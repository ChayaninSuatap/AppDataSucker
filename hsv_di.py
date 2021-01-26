from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from icon_util import open_and_resize

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Dropout, LeakyReLU, concatenate

def make_feature(fn, bin=10):
    img = Image.open(fn)
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
    input_layer = Input(shape=(input_size,))
    x = Dense(10)(input_layer)
    x = LeakyReLU()(x)
    x = Dense(5)(x)
    x = LeakyReLU()(x)
    x = Dense(2, activation='softmax')(x)

    def add_dense():
    
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    return model
    

if __name__ == '__main__':
    # feature = make_feature('allcolor.jpg', 5)
    # print(feature)

    m = make_model(10)
