
import matplotlib.pyplot as plt
import numpy as np
import itertools
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
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text = '0' if cm[i,j] == 0 else format(cm[i, j], '.1f')
        plt.text(j, i, text,
                 horizontalalignment="center", verticalalignment='center', fontsize='small',
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    # get inputs from excel or sheet. thesis only.
    a = []
    print('Please input:')
    for i in range(17):
        x = input()
        a.append([float(xx)*100/20 for xx in x.split('\t')])
    a = np.array(a) 

    
    _plot_confusion_matrix(a, ['Board', 'Trivia', 'Arcade', 'Card', 'Music', 'Racing',
 'Action', 'Puzzle', 'Simulation', 'Strategy', 'Role playing', 'Sports', 'Adventure', 'Casino', 'Word', 'Casual', 'Educational'],
 title=''
 )
