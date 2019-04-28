from plt_util import plot_confusion_matrix
import pickle
from keras.models import load_model
import random
import db_util
import icon_util
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
IS_REGRESSION = True
model_path = \
'armnet_regression-ep-002-loss-0.760-vloss-0.791-vmas-0.766.hdf5'

# prepare x y
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
random.seed(21)
np.random.seed(21)
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*80/100)
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    if IS_REGRESSION: rating = float(app_ids_and_labels[i][1])
    app_ids_and_labels[i] = app_id, rating
xs = []
ys = []
# split with myself
for app_id, label in app_ids_and_labels[ninety:]:
    try:
        icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
        xs.append(np.array(icon))
        ys.append(label)
    except:
        pass
xs = np.array(xs)
xs = xs.astype('float32')
xs /= 255
if not IS_REGRESSION:
    ys = to_categorical(ys, 4)
    plot_confusion_matrix(model_path, (xs,ys), 
        batch_size=64, fn_postfix='')
else:
    from keras.layers import Activation
    from keras import backend as K
    from keras.utils.generic_utils import get_custom_objects
    def my_sigmoid(x):
        return (K.sigmoid(x) * 5)
    act = Activation(my_sigmoid)
    act.__name__ = 'my_sigmoid'
    get_custom_objects().update({'my_sigmoid': act})
    model = load_model(model_path)
    pred = model.predict(xs, batch_size=64)    
    with open('pred.txt','w') as f:
        for x in pred:
            f.write(str(x[0]) + '\n')
    print(pred)



