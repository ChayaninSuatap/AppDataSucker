from plt_util import plot_confusion_matrix
import pickle
from keras.models import load_model
import random
import db_util
import icon_util
import numpy as np
from keras.utils import to_categorical
model_path = 'armnet_1.0-ex-10-attemp2-ep-018-loss-0.05-acc-0.94-vloss-4.91-vacc-0.39.hdf5'


# prepare x y
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
random.seed(21)
random.shuffle(app_ids_and_labels)
ninety = int(len(app_ids_and_labels)*80/100)
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
    batch_size=32, fn_postfix='armnet_fixed_ex11_attemp2_ep18')


