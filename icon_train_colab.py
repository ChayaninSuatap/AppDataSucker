from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from keras.models import load_model, save_model
from keras.callbacks import Callback
import math
import random
from sklearn.model_selection import train_test_split
import sqlite3
from PIL import Image
import os
#my lib
#icon_util
def load_icon_by_app_id(app_id, resizeW, resizeH):
    return open_and_resize('icons/' + app_id + '.png', resizeW, resizeH)

def open_and_resize(fn, resizeW, resizeH):
    return np.asarray( _convert_to_rgba(fn, resizeW, resizeH ))[:,:,:3]

def _convert_to_rgba(fn, resizeW, resizeH):
    png = Image.open(fn).convert('RGBA')
    png = png.resize( (resizeW, resizeH))
    background = Image.new('RGBA', png.size, (255,255,255))

    alpha_composite = Image.alpha_composite(background, png)
    return alpha_composite
#keras_util
def compute_class_weight(labels):
    #make class_freq
    class_freq={}
    for x in labels:
        if not (x in class_freq):
            #not already got key
            class_freq[x]=1
        else:
            class_freq[x]+=1
    #make class weight
    class_weight={}
    minfreq = min(v for k,v in class_freq.items())
    for k,v in class_freq.items():
        class_weight[k] = minfreq/v
    return class_weight
def group_for_fit_generator(xs, n, shuffle=False):
    i = 0
    out = []
    if shuffle : random.shuffle(xs)
    for x in xs:
        i+=1
        out.append(x)
        if i == n:
            i = 0
            yield out
            out = []
    if out != []:
        yield out
import matplotlib.pyplot as plt
def _get_weights_of_layers(model):
    output = []
    for layer in model.layers:
        w_and_b = layer.get_weights()
        if w_and_b != []:
            if len(w_and_b)>1: #bias valid
                weights = w_and_b[0].flatten()
                biases = w_and_b[1].flatten()
                output.append( np.append(weights,biases))
            else:
                weights = w_and_b[0].flatten()
                output.append( weights)
    return output
class PlotAccLossCallback(Callback):
    def on_train_begin(self, logs={}):
        self.log_loss = []
        self.log_vloss = []
        self.log_acc = []
        self.log_vacc = []
        self.log_weights = []
        grid = plt.GridSpec(2,2)
        self.fig = plt.figure()
        self.loss_plt = plt.subplot(grid[0,0])
        self.acc_plt = plt.subplot(grid[0,1])
        self.weights_plt = plt.subplot(grid[1,0:])
        self.weights_last_epoch = _get_weights_of_layers(self.model)
        plt.get_current_fig_manager().window.state('zoomed')
    def on_epoch_end(self, epoch, logs={}):
        #update data
        self.log_loss.append(logs['loss'])
        self.log_vloss.append(logs['val_loss'])
        self.log_acc.append(logs['acc'])
        self.log_vacc.append(logs['val_acc'])
        #update weights data
        weights_this_epoch = _get_weights_of_layers(self.model)
        diff_weights = []
        for old, new in zip(self.weights_last_epoch, weights_this_epoch):
            diff_weights.append(np.absolute(old-new).mean())
        self.log_weights.append(diff_weights)
        self.weights_last_epoch = weights_this_epoch
        #plot loss
        self.loss_plt.cla()
        self.loss_plt.plot(self.log_loss)
        self.loss_plt.plot(self.log_vloss)
        self.loss_plt.set(xlabel='epoch',ylabel='loss')
        self.loss_plt.legend(['train','test'], loc='upper left')
        self.loss_plt.set_title('loss')
        #plot acc
        self.acc_plt.cla()
        self.acc_plt.plot(self.log_acc)
        self.acc_plt.plot(self.log_vacc)
        self.acc_plt.set(xlabel='epoch',ylabel='acc')
        self.acc_plt.legend(['train','test'], loc='upper left')
        self.acc_plt.set_title('accuracy')
        #plot weights adjustment
        self.weights_plt.cla()
        npw = np.array(self.log_weights)
        legends=[]
        for ilayer in range(len(self.log_weights[0])):
            legends.append('layer '+str(ilayer+1))
            self.weights_plt.plot(npw[:,ilayer])
        self.weights_plt.legend(legends, loc='upper right') 

        self.weights_plt.set_title('weigts adjustment')
        plt.draw()
#db_util
def connect_db():
    return sqlite3.connect('data.db')
#train image
# get features_and_labels
conn = connect_db()
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
                    icon = load_icon_by_app_id(app_id, 128,128)
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
                    icon = load_icon_by_app_id(app_id, 128,128)
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
filepath='armnet_bn_after_relu_cont-{epoch:03d}-loss-{loss:.3f}-acc-{acc:.3f}-vloss-{val_loss:.3f}-vacc-{val_acc:.3f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=1, period=2)
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