import numpy as np
import db_util
import random
from keras_util import compute_class_weight, group_for_fit_generator, PlotAccLossCallback, create_image_data_gen
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU
from keras.models import Model
from datetime import datetime
import icon_util
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import math
from keras.metrics import categorical_accuracy
from keras import backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects

#pre setting
IS_REGRESSION = False
CLASS_NUM = 3


conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
#random shuffle 
time_seed = int(datetime.now().microsecond)
random.seed(time_seed)
random.seed(7)
np.random.seed(7)

print('time_seed',time_seed)
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*80/100)
#calculate label class
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.9: rating = 0
    elif float(rating) > 3.9 and float(rating) <= 4.4: rating = 1
    # elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 2
    if IS_REGRESSION: rating = float(app_ids_and_labels[i][1])
    app_ids_and_labels[i] = app_id, rating
#split train test
app_ids_and_labels_train = app_ids_and_labels[:ninety]
app_ids_and_labels_test = app_ids_and_labels[ninety:]

#oversample
# if not IS_REGRESSION: icon_util.oversample_image(app_ids_and_labels_train)

#class weight
if not IS_REGRESSION:
    class_weight = compute_class_weight(x for _,x in app_ids_and_labels_train)
    print(class_weight)
    #train test label distribution
    dist = {0:0,1:0,2:0}
    for _,x in app_ids_and_labels_train:
        dist[x]+=1
    print('train dist', dist)
    dist = {0:0,1:0,2:0}
    for _,x in app_ids_and_labels_test:
        dist[x]+=1
    print('test dist', dist)
else:
    pass
    # with open('train_rating.txt','w') as f:
    #     for _,x in app_ids_and_labels_train:
    #         f.write(str(x) + '\n')
    # input()
#make model
input_layer = Input(shape=(128, 128, 3))
# x = Conv2D(32,(3,3), name='my_model_conv_2')(input_layer)
# x = LeakyReLU()(x)
# x = Dropout(0.1)(x)
x = Conv2D(64,(3,3), name='my_model_conv_3')(input_layer)
x = LeakyReLU()(x)
x = Dropout(0.1)(x)
x = MaxPooling2D((2,2), name='my_model_max_pooling_2')(x)

x = Conv2D(128,(3,3), name='my_model_conv_4')(x)
x = LeakyReLU()(x)
x = Dropout(0.1)(x)
x = MaxPooling2D((2,2), name='my_model_max_pooling_4')(x)

x = Conv2D(256,(3,3), name='my_model_conv_5')(x)
x = LeakyReLU()(x)
x = Dropout(0.1)(x)

x = Conv2D(128,(1,1), name='my_model_conv_11')(x)
x = LeakyReLU()(x)
# x = MaxPooling2D((2,2), name='my_model_max_pooling_3')(x)

# x = MaxPooling2D((2,2), name='my_model_max_pooling_4')(x)
x = Flatten(name='my_model_flatten')(x)
# custom activation fn
def my_sigmoid(x): return (K.sigmoid(x)*5)
act = Activation(my_sigmoid)
act.__name__ = 'my_sigmoid'
get_custom_objects().update({'my_sigmoid':act})
#output layer
if IS_REGRESSION:
    x = Dense(1, activation='my_sigmoid', name='my_model_regress_1')(x)
else:
    x = Dense(16, name='my_model_dense_2')(x)
    x = LeakyReLU()(x)
    x = Dense(3, activation='softmax', name='my_model_dense_3')(x)
model = Model(input=input_layer, output=x)
#compile
if IS_REGRESSION:
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
else:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
#generator
epochs = 999
batch_size = 32

def generator():
    datagen = create_image_data_gen()
    for i in range(epochs):
        random.shuffle(app_ids_and_labels_train)
        for g in group_for_fit_generator(app_ids_and_labels_train, batch_size, shuffle=True):
            icons = []
            labels = []
            #prepare chrunk
            for app_id, label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
                    icons.append(icon)
                    labels.append(label)
                    # if i == 0 : print(app_id)
                except:
                    # print('icon error:', app_id)
                    pass

            icons = np.asarray(icons)

            #data gen
            # icons_for_gen = np.asarray(icons)
            # labels_for_gen = np.asarray(labels)
            # for xs, ys in datagen.flow(icons_for_gen,labels_for_gen, batch_size=icons_for_gen.shape[0]):
            #     icons = xs
            #     labels = ys
            #     break

            #normalize
            icons = icons.astype('float32')
            icons /= 255
            if IS_REGRESSION:
                labels = np.array(labels)
                # print(labels)
            else: labels = to_categorical(labels, 3)
            yield icons, labels
def test_generator():
    for i in range(epochs):
        for g in group_for_fit_generator(app_ids_and_labels_test, batch_size):
            icons = []
            labels = []
            #prepare chrunk
            for app_id, label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 128,128)
                    icons.append(icon)
                    labels.append(label)
                except:
                    # print('icon error:', app_id)
                    pass
            icons = np.asarray(icons)
            icons = icons.astype('float32')
            icons /= 255
            if IS_REGRESSION:
                labels = np.array(labels)
            else: 
                labels = to_categorical(labels, 3)
            yield icons, labels

# write save each epoch
filepath='armnet_3_class_init_filer_64-ep-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
if IS_REGRESSION:
    filepath='armnet_regression-ep-{epoch:03d}-loss-{loss:.3f}-vloss-{val_loss:.3f}-vmas-{val_mean_absolute_error:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0, period=1)
palc = PlotAccLossCallback()
# do it
history = model.fit_generator(generator(),
    steps_per_epoch=math.ceil(len(app_ids_and_labels_train)/batch_size),
    validation_data=test_generator(), max_queue_size=1,
    validation_steps=math.ceil(len(app_ids_and_labels_test)/batch_size),
    epochs=epochs , callbacks=[checkpoint, palc], verbose=1,
    class_weight=None if IS_REGRESSION else class_weight)