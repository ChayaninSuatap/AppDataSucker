from plt_util import plot_confusion_matrix
import pickle
from keras.models import load_model
import random
import db_util
import icon_util
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
model_path = \
'armnet_bn_after_relu_cont-166-loss-2.944-acc-0.528-vloss-7.437-vacc-0.539.hdf5'

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
    app_ids_and_labels[i] = app_id, rating
xs = []
ys = []
#split with sklearn
# app_ids = np.array([x[0] for x in app_ids_and_labels])
# labels = np.array([x[1] for x in app_ids_and_labels])
# xtrain, xtest, ytrain, ytest = train_test_split(app_ids, labels, test_size=0.2)
# for app_id, label in zip(xtest, ytest):
#     try:
#         icon = icon_util.load_icon_by_app_id(app_id, 244, 244)
#         xs.append(np.array(icon))
#         ys.append(label)
#     except:
#         pass

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
ys = to_categorical(ys, 4)
plot_confusion_matrix(model_path, (xs,ys), 
    batch_size=32, fn_postfix='')


