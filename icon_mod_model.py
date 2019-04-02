import keras
from keras.models import load_model, save_model
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D
from keras.models import Model

#make image model

# old_model = load_model('inception_v4.hdf5')
# old_model.layers.pop()
# t = old_model.layers[-1].output
# t = Dense(4, activation='softmax')(t)
# new_model = Model(inputs=old_model.input, outputs=t)
# new_model.compile(loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy'])
# save_model(new_model, 'model_modded.hdf5')

model = load_model('models_modded/densenet121.hdf5')
for layer in model.layers: layer.trainable = False
model.layers.pop() # pop dense 4
model.layers.pop() # pop global average
# conv_32_1_1_mod (1,1)
t = model.layers[-1].output
t = Conv2D(32, (1,1), activation='relu', name='conv2d_32_1_1_mod')(t)
t = GlobalAveragePooling2D()(t)
t = Dense(4, activation='relu', name='dense_4_mod')(t)
newmodel = Model(input=model.input, output=t)
newmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
print(newmodel.summary())
newmodel.save('models_modded/densenet_conv_32_1_1.hdf5')
# global avg pooling 
# dense 4

