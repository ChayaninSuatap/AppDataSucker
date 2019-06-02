import icon_util
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU
from keras.models import Model
from keras_util import group_for_fit_generator
import random
import numpy as np

def compute_baseline(aial):
    total = 0
    for _,x,_ in aial:
        total += x
    avg = total / len(aial)

    total_mse = 0
    for _,x,_ in aial:
        total_mse += (x-avg) ** 2
    return avg, total_mse/ len(aial)


def create_icon_cate_model():
    o = icon_util.create_model(IS_REGRESSION=True)
    input_layer = o['input_layer']
    flatten_layer = o['flatten_layer']
    output_layer = o['output_layer']

    #jump
    x = Dense(32)(flatten_layer)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output_cate = Dense(18, activation='sigmoid')(x)

    model = Model(input=input_layer, output=[output_layer, output_cate])
    model.compile(optimizer='adam',
        loss={'my_model_regress_1':'mse','dense_2':'binary_crossentropy'},
        metrics={'my_model_regress_1':'mean_absolute_percentage_error'})
    model.summary()
    return model

def datagenerator(aial, batch_size, epochs):

    for i in range(epochs):
        random.shuffle(aial)
        for g in group_for_fit_generator(aial, batch_size, shuffle=True):
            icons = []
            labels = []
            cate_labels = []
            #prepare chrunk
            for app_id, label, cate_label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
                    icons.append(icon)
                    labels.append(label)
                    cate_labels.append(cate_label)
                except:
                    pass

            icons = np.asarray(icons)

            #normalize
            icons = icons.astype('float32')
            icons /= 255
            labels = np.array(labels)
            cate_labels = np.array(cate_labels)
            yield icons, [labels, cate_labels]



