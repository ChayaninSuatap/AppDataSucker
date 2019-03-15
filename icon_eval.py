from plt_util import plot_confusion_matrix
import pickle
from keras.models import load_model
import random
import db_util
import icon_util
import numpy as np
from keras.utils import to_categorical
model_path = 'densenet-ep-4.hdf5'


# prepare x y
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
random.seed(21)
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*90/100)
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    app_ids_and_labels[i] = app_id, rating
app_ids_and_labels = app_ids_and_labels[ninety:]
xs = []
ys = []
for app_id, label in app_ids_and_labels:
    try:
        icon = icon_util.load_icon_by_app_id(app_id, 244, 244)
        xs.append(np.array(icon))
        ys.append(label)
    except:
        pass
xs = np.array(xs)
plot_confusion_matrix(model_path, '', is_regression=False,
    batch_size=32, no_xy_path=True, xy_obj=(xs,ys), fn_postfix='09')

