import numpy as np
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

input_layer = Input(shape=(100,))
x = Dense(100)(input_layer)
x = Dense(1000)(x)
x = Dense(1000)(x)
output_layer = Dense(100)(x)

xs = np.random.rand(10000,100)
ys = np.random.rand(10000,100)

model = Model(inputs = input_layer, outputs = output_layer)
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit(xs, ys, validation_split=0.2, epochs=999, batch_size=64)

