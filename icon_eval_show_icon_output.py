import numpy as np
from keras.models import load_model
import db_util
import icon_util
from shutil import  copyfile
import matplotlib.pyplot as plt
import random

model = load_model('armnet_3_class_4_layers-ep-080-loss-0.033-acc-0.980-vloss-3.774-vacc-0.429.hdf5')
conn = db_util.connect_db()
app_ids_and_labels = []
dat=conn.execute('select app_id, rating from app_data')
for x in dat:
    if x[1] != None:
      app_ids_and_labels.append( (x[0], x[1]))  
#random shuffle 

random.seed(7)
np.random.seed(7)

#split class range
for i in range(len(app_ids_and_labels)):
    app_id , rating = app_ids_and_labels[i]
    if float(rating) <= 3.9: rating = 0
    elif float(rating) > 3.9 and float(rating) <= 4.4: rating = 1
    else: rating = 2
    app_ids_and_labels[i] = app_id, rating
ninety = int(len(app_ids_and_labels)*80/100)
random.shuffle(app_ids_and_labels)
app_ids_and_labels_test = app_ids_and_labels[ninety:]
#print test dist
dist = {0:0,1:0,2:0,3:0}
for _,x in app_ids_and_labels[ninety:]:
    dist[x]+=1
print('test dist', dist)
#confusion matrix
conmat = [[0,0,0],[0,0,0],[0,0,0]]
#loop
for app_id, label in app_ids_and_labels_test:
    try:
        icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
    except:
        continue
    icons = np.array([icon])
    icons = icons.astype('float32')
    icons /= 255
    pred = model.predict(icons)
    pred_label = np.argmax(pred[0])
    conmat[label][pred_label] += 1
    fn = 'icons_pred/%d_%d/%d_%d_%s.png' % (label, pred_label, label, pred_label, app_id)
    copyfile('icons/%s.png' % (app_id,), fn)
print(conmat)


