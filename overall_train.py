import overall_db_util
from overall_feature_util import extract_feature_vec
import random
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.utils.np_utils import to_categorical

#extract feature vector
vec_and_labels = []
for record in overall_db_util.query() :
    t = extract_feature_vec(record, use_download_amount=False, use_rating_amount=False)
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
output_shape = len( set(label))

#separate train , test
ninety = int(len(vec_and_labels) * 90 / 100)
x90 = np.asarray(featvec[:ninety])
x10 = np.asarray(featvec[ninety:])
y90 = to_categorical(np.asarray(label[:ninety]), output_shape)
y10 = to_categorical(np.asarray(label[ninety:]), output_shape)

#make model
input_shape = len(x90[0])
dense_size = input_shape
print('input_shape :', input_shape)

layer_input = Input(shape=(input_shape,), name='overall_input')
t = Dense( dense_size, activation='relu' )(layer_input)
for i in range(10) :
    t = Dense(dense_size, activation='relu')(t)
layer_output = Dense(2, activation='softmax', name='overall_output')(t)

model = Model(inputs = layer_input, outputs = layer_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x90, y90, validation_data=(x10, y10), epochs=10, batch_size=32)    