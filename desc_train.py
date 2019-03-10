import db_util
import re
import random
import string
import pickle
import desc_util
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import global_util as util
conn = db_util.connect_db()
dat = conn.execute('select description, rating, app_id from app_data where not (rating is NULL) and not (description is NULL)')
dat = list(dat)
random.seed(21)
random.shuffle(dat)
descs = [x[0] for x in dat]
labels = [x[1] for x in dat] # rating floating point
#convert floating point to discrete
for i,rating in enumerate(labels):
    if float(rating) <= 3.5: rating = 0
    elif float(rating) > 3.5 and float(rating) <= 4.0: rating = 1
    elif float(rating) > 4.0 and float(rating) <= 4.5: rating = 2
    else: rating = 3
    labels[i] = rating
labels = to_categorical(labels, 4)

# with open('descs.obj','rb') as f: english_only_descs = pickle.load(f)
# filter only english
descs, labels = desc_util.preprocess_text(descs, labels, 0.7)
# with open('descs.obj','wb') as f: pickle.dump(english_only_descs,f)

indexized_words = desc_util.indexize_words(descs)
print(indexized_words, len(indexized_words[0]))
sequence_size = len(indexized_words[0])

num_words = 2500
embed_dim = 128
lstm_out = 16

input_layer = Input(shape = (sequence_size,))
x = Embedding(input_dim=num_words, output_dim=embed_dim, input_length=sequence_size)(input_layer)
x = LSTM(lstm_out)(x)
output_layer = Dense(4, activation='softmax')(x)

model = Model(input=input_layer, output=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
fn = 'descs_ep-{epoch:03d}-loss-{loss:.2f}-acc-{acc:.2f}-val_loss-{val_loss:.2f}-val_acc-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(fn, save_best_only=False)

class SaveHistory(Callback):
    def on_epoch_end(self, batch, log={}):
        util.save_pickle(self.model.history , 'model_history.obj')

# save_history = SaveHistory()

model.fit(x=indexized_words, y=labels, batch_size=32, validation_split=0.1,
    epochs=999, class_weight={1: 0.314, 2: 0.125, 3: 0.3, 0: 1.0},
    callbacks=[checkpoint])




