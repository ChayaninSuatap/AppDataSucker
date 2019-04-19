import icon_util
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import db_util
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
import math
import random
from keras_util import compute_class_weight, group_for_fit_generator, PlotAccLossCallback
from sklearn.model_selection import train_test_split
#train image
# get features_and_labels
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
random.seed(21)
np.random.seed(21)
#calculate label class
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    app_ids_and_labels[i] = app_id, rating
ninety = int(len(app_ids_and_labels)*80/100)
#class weight
class_weight = compute_class_weight(x for _,x in app_ids_and_labels)
print(class_weight)

#isolate app_ids and labels
# app_ids = np.array([x[0] for x in app_ids_and_labels])
# labels = np.array([x[1] for x in app_ids_and_labels])
# xtrain, xtest, ytrain, ytest = train_test_split(app_ids, labels, test_size=0.2)
# load pretrain model or get the old one
model = load_model('armnet_bn_after_relu-ep-055-loss-2.92-acc-0.32-vloss-11.00-vacc-0.32.hdf5')

# write fit generator
epochs = 999
batch_size = 48
def generator():
    for i in range(epochs):
        # for g in group_for_fit_generator(list(zip(xtrain,ytrain)), batch_size, shuffle=True):
        for g in group_for_fit_generator(app_ids_and_labels[:ninety], batch_size, shuffle=True):
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
            labels = to_categorical(labels, 4)
            yield icons, labels

def test_generator():
    for i in range(epochs):
        # for g in group_for_fit_generator(zip(xtest,ytest), batch_size):
        for g in group_for_fit_generator(app_ids_and_labels[ninety:], batch_size):
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
            labels = to_categorical(labels, 4)
            yield icons, labels

# write save each epoch
filepath='armnet_bn_after_relu_cont-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=1)
palc = PlotAccLossCallback()
# fit train test split
# history = model.fit_generator(generator(),
#     steps_per_epoch=math.ceil(xtrain.shape[0]/batch_size),
#     validation_data=test_generator(), max_queue_size=1,
#     validation_steps=math.ceil(xtest.shape[0]/batch_size),
#     epochs=epochs , callbacks=[checkpoint, palc], verbose=1, class_weight=class_weight, initial_epoch=55)
# fit split by myself
history = model.fit_generator(generator(),
    steps_per_epoch=math.ceil(len(app_ids_and_labels[:ninety])/batch_size),
    validation_data=test_generator(), max_queue_size=1,
    validation_steps=math.ceil(len(app_ids_and_labels[ninety:])/batch_size),
    epochs=epochs , callbacks=[checkpoint, palc], verbose=1, class_weight=class_weight)#, initial_epoch=55)