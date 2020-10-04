import random
import numpy as np
import icon_util
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import math

class image_batch_sequence(Sequence):

    def __init__(self, data, batch_size, resize_size=(128,128), datagen=None, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.datagen = datagen
        self.shuffle = shuffle
    
    def __len__(self):
        return math.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, idx):
        start_idx  = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))

        xs = []
        yss = [[] for i in range(len(self.data[0])-1)]

        for i in range(start_idx, end_idx):
            row = self.data[i]
            img = icon_util.load_icon_by_app_id(row[0], self.resize_size[0], self.resize_size[1])
            xs.append(img)
            for y_col_i, y_col in enumerate(row[1:]):
                yss[y_col_i].append(y_col)

        xs = np.array(xs)
        if self.datagen:
            for augmented_chrunk in self.datagen.flow(xs, batch_size=xs.shape[0], shuffle=False):
                xs = augmented_chrunk
                break
        
        
        xs = xs.astype('float32')/255
        yss_np = []
        for ys in yss:
            yss_np.append(np.array(ys)) 
        if len(yss_np) == 1:
            return (xs, yss_np[0])
        else:
            return (xs, yss_np[1])
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)




        