import icon_util
import numpy as np
import db_util
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras_util import PlotWeightsCallback

#query
conn = db_util.connect_db()
dat = conn.execute('select app_id, category from app_data')
array = []
for x in dat: #remove null
    if x[1] != None:
        # assign class label
        label = 1 if 'CASINO' in x[1] else 0
        array.append( (x[0] , label))
# random.seed(21)
random.shuffle(array)
# np.random.seed(21)
df = pd.DataFrame(array, columns=['x','y'])

#sample non casino with # equal casino
df_casino = df[df.y==1]
casino_n = len(df_casino)
df_non_casino = df[df.y!=1].sample(casino_n)
df = pd.concat([df_casino, df_non_casino])
#split
X = []
Y = []
for x,y in zip(df.x.values, df.y.values):
    try:
        icon = icon_util.load_icon_by_app_id(x, 128, 128)
        X.append(icon)
        Y.append(y)
    except:
        pass
X = np.array(X)
Y = to_categorical(Y,2)
xtrain, xtest,  ytrain, ytest = train_test_split(X, Y, test_size=0.2)

input_layer = Input(shape=(128, 128, 3))
x = Conv2D(32,(3,3), activation='relu', name='my_model_conv_1', kernel_initializer='glorot_uniform')(input_layer)
# x = MaxPooling2D((2,2), name='my_model_max_pooling_1')(x)
# x = Dropout(0.1, name='my_model_dropout_1')(x)
x = Conv2D(64,(3,3), activation='relu', name='my_model_conv_2', kernel_initializer='glorot_uniform')(x)
x = MaxPooling2D((2,2), name='my_model_max_pooling_2')(x)
# x = Dropout(0.1, name='my_model_dropout_2')(x)
# x = Conv2D(32,(3,3), activation='relu', name='my_model_conv_3', kernel_initializer='glorot_uniform')(x)
# x = MaxPooling2D((2,2), name='my_model_max_pooling_3')(x)
# x = Dropout(0.1, name='my_model_dropout_3')(x)
# x = Conv2D(64,(3,3), activation='relu', name='my_model_conv_4', kernel_initializer='glorot_uniform')(x)
x = Flatten(name='my_model_flatten')(x)
x = Dense(2, activation='softmax', name='my_model_dense_1', kernel_initializer='glorot_uniform')(x)
# x = Dense(, activation='softmax', name='my_model_dense_2', kernel_initializer='glorot_uniform')(x)
model = Model(input=input_layer, output=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# filepath='casino-ep-{epoch:03d}-loss-{loss:.2f}-acc-{acc:.2f}-vloss-{val_loss:.2f}-vacc-{val_acc:.2f}.hdf5'
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, verbose=0)
pwc = PlotWeightsCallback()
model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=999, batch_size=32, callbacks=[pwc])
# model.fit(xtrain, ytrain, validation_data=(xtest,ytest), epochs=999, batch_size=32,callbacks=[checkpoint])




