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

def group_for_fit_generator(xs, n):
    i = 0
    out = []
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
            weights = w_and_b[0].flatten()
            biases = w_and_b[1].flatten()
            output.append( np.append(weights,biases))
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



