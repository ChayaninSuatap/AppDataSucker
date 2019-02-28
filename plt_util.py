import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from overall_util import save_prediction_to_file
import pickle
def plot_loss(history, is_regression):
    if not is_regression:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    else:
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graph_acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graph_loss.png')
    plt.clf()

def plot_confusion_matrix(model_path, xy_path, is_regression, batch_size, fn_postfix=''):
    #def my_sigmoid
    def my_sigmoid(x):
        return (K.sigmoid(x) * 5)
    act = Activation(my_sigmoid)
    act.__name__ = 'my_sigmoid'
    get_custom_objects().update({'my_sigmoid': act})

    #load model
    model = load_model(model_path)
    #load pickle
    with open(xy_path, 'rb') as f:
        x,y = pickle.load(f)
        x = [x[0], x[1], x[2], x[3]]
    if is_regression:
        #regression
        pass
    else:
        #vars
        batch_size = 32
        #classify 4 class
        y_10_eval = to_categorical(y, 4)
        print('evaluating')
        result = model.evaluate(x, y_10_eval, batch_size=batch_size)
        print(result)
        pred = model.predict(x, batch_size=batch_size)
        conmat = confusion_matrix(y, pred.argmax(axis=1))
        _plot_confusion_matrix(conmat, ['0 - 3.5','3.5 - 4','4 - 4.5','4.5 - 5'], fn_postfix=fn_postfix)
        _plot_confusion_matrix(conmat, ['0 - 3.5','3.5 - 4','4 - 4.5','4.5 - 5'], normalize=True, fn_postfix=fn_postfix)

def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          fn_postfix=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if normalize:
        plt.savefig('models/cm_norm' + fn_postfix + '.png')
    else:
        plt.savefig('models/cm' + fn_postfix + '.png')
    plt.clf()