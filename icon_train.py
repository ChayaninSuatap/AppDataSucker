from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
import math
import random
from sklearn.model_selection import train_test_split
#my lib
import icon_util
from keras_util import compute_class_weight, group_for_fit_generator, PlotAccLossCallback
import db_util
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
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*80/100)
#class weight
class_weight = compute_class_weight(x for _,x in app_ids_and_labels)
print(class_weight)

#isolate app_ids and labels
# app_ids = np.array([x[0] for x in app_ids_and_labels])
# labels = np.array([x[1] for x in app_ids_and_labels])
# xtrain, xtest, ytrain, ytest = train_test_split(app_ids, labels, test_size=0.2)
# load pretrain model or get the old one
model = load_model('armnet_dropout_0.1-ep-030-loss-0.077-acc-0.888-vloss-3.832-vacc-0.393.hdf5')

# write fit generator
epochs = 999
batch_size = 32
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
                    icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
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

#train test label distribution
dist = {0:0,1:0,2:0,3:0}
for _,x in app_ids_and_labels[:ninety]:
    dist[x]+=1
print('train dist', dist)
dist = {0:0,1:0,2:0,3:0}
for _,x in app_ids_and_labels[ninety:]:
    dist[x]+=1
print('test dist', dist)
# write save each epoch
filepath='armnet_dropout_0.1-ep-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
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
    epochs=epochs , callbacks=[checkpoint, palc], verbose=1, class_weight=class_weight, initial_epoch=30)