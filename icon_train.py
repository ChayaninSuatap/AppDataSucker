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
from keras_util import compute_class_weight, group_for_fit_generator

#train image
# get features_and_labels
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
random.seed(21)
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*90/100)
#calculate label class
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    app_ids_and_labels[i] = app_id, rating
#class weight
class_weight = compute_class_weight(x for _,x in app_ids_and_labels)
print(class_weight)

# load pretrain model or get the old one
model = load_model('model_modded.hdf5')
# write fit generator

epochs = 999
batch_size = 32 
def generator():
    for i in range(epochs):
        for g in group_for_fit_generator(app_ids_and_labels[:ninety], batch_size):
            icons = []
            labels = []
            #prepare chrunk
            for app_id, label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 244,244)
                    icons.append(icon)
                    labels.append(label)
                except:
                    # print('icon error:', app_id)
                    pass
            icons = np.asarray(icons)
            labels = to_categorical(labels, 4)
            yield icons, labels

def test_generator():
    for i in range(epochs):
        for g in group_for_fit_generator(app_ids_and_labels[ninety:], batch_size):
            icons = []
            labels = []
            #prepare chrunk
            for app_id, label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 244,244)
                    icons.append(icon)
                    labels.append(label)
                except:
                    # print('icon error:', app_id)
                    pass
            icons = np.asarray(icons)
            labels = to_categorical(labels, 4)
            yield icons, labels

# write save each epoch
filepath='model-{epoch:03d}-{loss:.3f}-{acc:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=1)
# do it
history = model.fit_generator(generator(),
    steps_per_epoch=math.ceil(len(app_ids_and_labels[:ninety])/batch_size),
    validation_data=test_generator(),
    validation_steps=math.ceil(len(app_ids_and_labels[ninety:])/batch_size),
    epochs=epochs , callbacks=[checkpoint], verbose=1, class_weight=class_weight)
