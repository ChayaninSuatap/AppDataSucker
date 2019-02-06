import overall_db_util
from overall_feature_util import extract_feature_vec
import random
from keras.layers import Input, Dense, concatenate
from keras.models import Model
import numpy as np
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

#config
USE_CONTENT_RATING = True

#extract feature vector
vec_and_labels = []
for record in overall_db_util.query() :
    t = extract_feature_vec(record, use_content_rating=USE_CONTENT_RATING,use_rating_amount=True, use_download_amount=True)
    vec_and_labels.append(t)

#shuffle
random.seed(1)
random.shuffle(vec_and_labels)

#separate vec and label
featvec = []
featvec_content_rating = []
label = []
for x in vec_and_labels :
    if USE_CONTENT_RATING:
        featvec.append( x[0][:-38])
        featvec_content_rating.append( x[0][-38:])
    else:
        featvec.append( x[0])
    label.append( x[1])
print('train y ratio :', label.count(0), label.count(1))
output_shape = len( set(label))

#separate train , test
ninety = int(len(vec_and_labels) * 90 / 100)
x90 = np.asarray(featvec[:ninety])
x90cr = np.asarray(featvec_content_rating[:ninety])
x10 = np.asarray(featvec[ninety:])
x10cr = np.asarray(featvec_content_rating[ninety:])
print('test y ratio :', label[ninety:].count(0), label[ninety:].count(1))
y90 = np.asarray([[x] for x in label[:ninety]])
y10 = np.asarray([[x] for x in label[ninety:]])

#custom activation
def my_sigmoid(x):
    return (K.sigmoid(x) * 5)
get_custom_objects().update({'my_sigmoid': Activation(my_sigmoid)})

#make model
input_shape = len(x90[0])
dense_size = input_shape
if USE_CONTENT_RATING:
    dense_size = input_shape + 10
print('input_shape :', input_shape)

if USE_CONTENT_RATING:
    content_rating_input = Input(shape=(38,), name='content_rating_input')
    content_rating_dense = Dense( 10, activation='relu')( content_rating_input)

    layer_input = Input(shape=(input_shape,), name='overall_input')
    if input_shape == 0:
        t = Dense( dense_size, activation='relu' )(content_rating_dense)
    else:
        t = concatenate([layer_input, content_rating_dense], name='content_rating_concat')
        t = Dense( dense_size, activation='relu' )(layer_input)

    for i in range(10) :
        t = Dense(dense_size, activation='relu')(t)
    
    layer_output = Dense(1, activation='my_sigmoid', name='overall_output')(t)

    if input_shape == 0:
        model = Model(inputs = content_rating_input, outputs = layer_output)
    else:
        model = Model(inputs = [layer_input,content_rating_input], outputs = layer_output)
else:
    layer_input = Input(shape=(input_shape,), name='overall_input')
    t = Dense( dense_size, activation='relu' )(layer_input)
    for i in range(10) :
        t = Dense(dense_size, activation='relu')(t)
    layer_output = Dense(1, activation='my_sigmoid', name='overall_output')(t)
    model = Model(inputs = layer_input, outputs = layer_output)


model.compile(optimizer='adam', loss='mse', metrics=['MAE','MSE'])

if USE_CONTENT_RATING and input_shape > 0:
    model.fit([x90,x90cr], y90, validation_data=([x10,x10cr], y10), epochs=20, batch_size=32)    
elif USE_CONTENT_RATING and input_shape == 0:
    model.fit(x90cr, y90, validation_data=(x10cr, y10), epochs=20, batch_size=32)    
else:
    model.fit(x90, y90, validation_data=(x10, y10), epochs=20, batch_size=32)

answers = [x[0] for x in model.predict([x10,x10cr], batch_size=32)]
f = open('answer.txt', 'w')
for x in answers:
    f.write(str(x)+'\n')