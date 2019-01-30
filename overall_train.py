import overall_db_util
from overall_feature_util import extract_feature_vec
import random
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

#extract feature vector
vec_and_labels = []
for record in overall_db_util.query() :
    t = extract_feature_vec(record, use_content_rating=False, use_category=False, use_app_version=False, use_in_app_products=False, use_sdk_version=True, use_screenshots_amount=False, use_rating_amount=True, use_last_update_date=False, use_price=False, use_download_amount=True)
    vec_and_labels.append(t)

#shuffle
random.seed(1)
random.shuffle(vec_and_labels)

#separate vec and label
featvec = []
label = []
for x in vec_and_labels :
    featvec.append( x[0])
    label.append( x[1])
print('train y ratio :', label.count(0), label.count(1))
output_shape = len( set(label))

#separate train , test
ninety = int(len(vec_and_labels) * 90 / 100)
x90 = np.asarray(featvec[:ninety])
x10 = np.asarray(featvec[ninety:])
print('test y ratio :', label[ninety:].count(0), label[ninety:].count(1))
y90 = np.asarray([[x] for x in label[:ninety]])
y10 = np.asarray([[x] for x in label[ninety:]])

#make model
input_shape = len(x90[0])
dense_size = input_shape
print('input_shape :', input_shape)

layer_input = Input(shape=(input_shape,), name='overall_input')
t = Dense( dense_size, activation='relu' )(layer_input)
for i in range(10) :
    t = Dense(dense_size, activation='relu')(t)

#custom activation
def my_sigmoid(x):
    return (K.sigmoid(x) * 5)
get_custom_objects().update({'my_sigmoid': Activation(my_sigmoid)})

layer_output = Dense(1, activation='my_sigmoid', name='overall_output')(t)

model = Model(inputs = layer_input, outputs = layer_output)
model.compile(optimizer='adam', loss='mse', metrics=['MAE','MSE'])

model.fit(x90, y90, validation_data=(x10, y10), epochs=20, batch_size=32)    