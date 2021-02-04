import os
import random
import icon_util
from sklearn.model_selection import KFold
import functools
import keras
from plt_util import plot_confusion_matrix
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import numpy as np
import math
from datetime import datetime
import global_util
from sklearn.preprocessing import  StandardScaler

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
    if shuffle:
        random.seed(datetime.now())
        random.shuffle(xs)
    for x in xs:
        i+=1
        out.append(x)
        if i == n:
            i = 0
            yield out
            out = []
    if len(out) > 0:
        yield out

class PlotConfusionMatrixCallback(Callback):
    def set_postfix_name(self, name):
        self.postfix_name = name

    def on_epoch_end(self, epoch, logs=None):
        x_test = self.validation_data[0]
        y_test = self.validation_data[1]
        plot_confusion_matrix(self.model, (x_test,y_test), 32,
            fn_postfix=self.postfix_name + '_ep_' + str(epoch+1), shut_up=True)

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


def _get_best_ep_and_acc(training_log):
    for i,x in enumerate(training_log):
        if i == 0:
            best_acc = x
            best_ep = i + 1
        elif best_acc < x:
            best_acc = x
            best_ep = i + 1
    return best_ep, best_acc

def get_initial_epoch(proj_path):
    pass

class SaveBestEpochCallback(Callback):
    def __init__(self, proj_path):
        self.proj_path = proj_path
        
        if not os.path.exists(proj_path+'training_log.obj'):
            global_util.save_pickle([])
        else:
            try:
                training_log = global_util.load_pickle(proj_path + 'training_log.obj')
            except:
                raise Exception('Can not load training_log.obj')

    def on_epoch_end(self, epoch, logs={}):


        try:
            old_log = global_util.load_pickle(proj_path + 'training_log.obj')
        except:
            raise Exception('Can not load training_log.obj')

        training_log = old_log[:]
        old_best_ep, old_best_acc = get_best_ep_and_acc(old_log)

        if logs['val_acc'] > old_best_acc:
            self.model.save(proj_path + 'best_model.hdf5')
            self.model.save(proj_path + 'best_model.backup.hdf5')
        
        training_log.append(logs['val_acc'])
        
        global_util.save_pickle(training_log, proj_path + 'training_log.obj')
        global_util.save_pickle(training_log, proj_path + 'training_log.backup.obj')

        self.model.save(proj_path + 'model.hdf5')
        self.model.save(proj_path + 'model.backup.hdf5')

class PlotWeightsCallback(Callback):
    def on_train_begin(self, logs={}):
        self.weights_last_epoch = _get_weights_of_layers(self.model)
    
    def on_epoch_end(self, epoch, logs={}):
        weights_this_epoch = _get_weights_of_layers(self.model)
        diff_weights = []
        for old, new in zip(self.weights_last_epoch, weights_this_epoch):
            diff_weights.append(np.absolute(old-new).mean())
        print('weight diff',diff_weights)
        self.weights_last_epoch = weights_this_epoch

import matplotlib.pyplot as plt
import matplotlib
class PlotAccLossCallback(Callback):
    def __init__(self, is_regression=False, is_cate=False, use_colab=False, proj='', use_paperspace=False):
        
        #is_cate should be False in every case.

        self.is_regression = is_regression
        self.is_cate = is_cate
        self.use_colab = use_colab
        self.use_paperspace = use_paperspace
        self.proj = proj
        from matplotlib.pyplot import rcParams
        rcParams['figure.figsize'] = 14, 8

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
    
    def on_epoch_end(self, epoch, logs={}):
        #update data
        
        if not self.is_regression and not self.is_cate :
            self.log_loss.append(logs['loss'])
            self.log_vloss.append(logs['val_loss'])
            self.log_acc.append(logs['acc'])
            self.log_vacc.append(logs['val_acc'])
        elif self.is_regression:
            self.log_loss.append(logs['loss'])
            self.log_vloss.append(logs['val_loss'])
            self.log_acc.append(logs['mean_absolute_percentage_error'])
            self.log_vacc.append(logs['val_mean_absolute_percentage_error'])
        elif self.is_cate:
            self.log_loss.append(logs['my_model_regress_1_loss'])
            self.log_vloss.append(logs['val_my_model_regress_1_loss'])
            self.log_acc.append(logs['my_model_regress_1_mean_absolute_percentage_error'])
            self.log_vacc.append(logs['val_my_model_regress_1_mean_absolute_percentage_error'])

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
        self.loss_plt.set_title('loss ep %d' % (epoch+1,))
        #plot acc
        self.acc_plt.cla()
        self.acc_plt.plot(self.log_acc)
        self.acc_plt.plot(self.log_vacc)
        self.acc_plt.set(xlabel='epoch',ylabel='acc')
        self.acc_plt.legend(['train','test'], loc='upper left')
        if self.is_regression or self.is_cate:
            self.acc_plt.set_title('mae percentage ep %d' % (epoch+1,))
        else: self.acc_plt.set_title('accuracy ep %d' % (epoch+1,))
        #plot weights adjustment
        self.weights_plt.cla()
        npw = np.array(self.log_weights)
        legends=[]
        for ilayer in range(len(self.log_weights[0])):
            legends.append('layer '+str(ilayer+1))
            self.weights_plt.plot(npw[:,ilayer])
        self.weights_plt.legend(legends, loc='upper right') 

        self.weights_plt.set_title('weigts adjustment ep %d' % (epoch+1,))
        
        #save fig
        if self.use_colab:
            fig_name = '/content/drive/My Drive/%s/%.03d.png' % (self.proj, epoch+1,)
            plt.savefig(fig_name)
        elif self.use_paperspace:
            fig_name = '/storage/%s/%.03d.png' % (self.proj, epoch+1,)
            plt.savefig(fig_name)
        else:
            fig_name = 'plots/%.03d.png' % (epoch+1,)
            if os.path.isfile(fig_name):
                os.remove(fig_name)
            plt.savefig(fig_name)
            plt.draw()
            plt.pause(0.01)

from keras.preprocessing.image import ImageDataGenerator

def create_image_data_gen():
    datagen = ImageDataGenerator(
    channel_shift_range=15,
    rotation_range=7,
    width_shift_range=0.04,
    height_shift_range=0.04,
    zoom_range=0.2
    )
    return datagen

def gen_k_fold_pass(aial, kf_pass, n_splits):
    app_ids_and_labels_train = []
    app_ids_and_labels_test = []
    kf = KFold(n_splits=n_splits, shuffle=False)
    kf_pass = kf_pass
    for i,(train_idxs, test_idxs) in enumerate(kf.split(aial)):
        if i == kf_pass:
            for idx in train_idxs:
                app_ids_and_labels_train.append( aial[idx])
            for idx in test_idxs:
                app_ids_and_labels_test.append( aial[idx])
            break
    return app_ids_and_labels_train, app_ids_and_labels_test

def metric_top_k(k):
    o = functools.partial(keras.metrics.top_k_categorical_accuracy, k=k)
    o.__name__ = 'top_' + str(k)
    return o

def metric_top_k_tf(k):
    o = functools.partial(keras.metrics.top_k_categorical_accuracy, k=k)
    o.__name__ = 'top_' + str(k)
    return o

def eval_top_5(model, test_generator, steps, use_tf_metric=False):
    gen = test_generator
    metric = metric_top_k_tf if use_tf_metric else metric_top_k
    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=[metric(1),metric(2),metric(3),metric(4), metric(5)])
    print(model.evaluate_generator(gen, steps=steps))

def plot_confusion_matrix_generator(model, ground_truth, test_gen_for_predict, steps_per_epoch):
    from sklearn.metrics import confusion_matrix
    from plt_util import _plot_confusion_matrix
    #process ground truth
    ys = np.array(ground_truth)
    y = [x.argmax() for x in ys]
    #predict
    pred = model.predict_generator(test_gen_for_predict, steps_per_epoch)
    #plot confusion
    conmat = confusion_matrix( y, pred.argmax(axis=1))
    _plot_confusion_matrix(conmat, [])

def standard_normalize(train_features, test_features):
    scaler = StandardScaler()
    fitted_scaler = scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    test_features = scaler.transform(test_features)

    return train_features, test_features, fitted_scaler