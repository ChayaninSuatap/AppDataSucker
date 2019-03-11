import keras
from keras.models import load_model, save_model
from keras.layers import Dense
from keras.models import Model

#make image model

old_model = load_model('inception_v4.hdf5')
old_model.layers.pop()
t = old_model.layers[-1].output
t = Dense(4, activation='softmax')(t)
new_model = Model(inputs=old_model.input, outputs=t)
new_model.compile(loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
save_model(new_model, 'model_modded.hdf5')

