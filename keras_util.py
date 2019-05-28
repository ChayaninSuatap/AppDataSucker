import os
import random
import icon_util
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

from plt_util import plot_confusion_matrix
from keras.callbacks import Callback
import numpy as np
import math
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
class PlotAccLossCallback(Callback):
    def __init__(self, is_regression=False, is_cate=False, use_colab=False, proj=''):
        self.is_regression = is_regression
        self.is_cate = is_cate
        self.use_colab = use_colab
        self.proj = proj
        from matplotlib.pyplot import rcParams
        rcParams['figure.figsize'] = 14, 8
        # if not self.use_colab:
            # plt.get_current_fig_manager().window.state('zoomed')

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
    width_shift_range=0.035,
    height_shift_range=0.035,
    )
    return datagen

# def split_class(app_ids_and_labels, class_num):



