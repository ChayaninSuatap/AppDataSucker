import random
import numpy as np
import icon_util
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
import math

class image_batch_sequence(Sequence):

    def __init__(self, data, batch_size, resize_size=(128,128), datagen=None, shuffle=True, app_id_overall_feature_d=None, overall_other_scaler=None):
        self.data = data
        self.batch_size = batch_size
        self.resize_size = resize_size
        self.datagen = datagen
        self.shuffle = shuffle
        self.app_id_overall_feature_d = app_id_overall_feature_d
        self.use_overall = self.app_id_overall_feature_d is not None
        self.overall_other_scaler = overall_other_scaler
    
    def __len__(self):
        return math.ceil(len(self.data)/self.batch_size)

    def __getitem__(self, idx):
        start_idx  = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.data))

        xs = []
        ys = []
        cates = []
        sdk_versions = []
        content_ratings = []
        others = []
        labels = []

        for i in range(start_idx, end_idx):
            row = self.data[i]
            img = icon_util.load_icon_by_app_id(row[0], self.resize_size[0], self.resize_size[1])
            xs.append(img)

            if self.use_overall:
                cate, sdk_version, content_rating, other, label = self.app_id_overall_feature_d[row[0]]
                cates.append(cate)
                sdk_versions.append(sdk_version)
                content_ratings.append(content_rating)
                others.append(other)
                labels.append(label)
                ys.append(row[1])
                if not (label == row[1]).all():
                    print(label)
                    print(row[1])
                    raise ValueError('app id may be mismatched')

        xs = np.array(xs)
        if self.datagen:
            for augmented_chrunk in self.datagen.flow(xs, batch_size=xs.shape[0], shuffle=False):
                xs = augmented_chrunk
                break
        
        
        xs = xs.astype('float32')/255
        cates = np.array(cates)
        sdk_versions = np.array(sdk_versions)
        content_ratings = np.array(content_ratings)
        others = np.array(others)
        ys = np.array(ys)

        #normalize others
        if self.overall_other_scaler is not None:
            others = self.overall_other_scaler.transform(others)

        if self.app_id_overall_feature_d is not None:
            return ([xs, cates, sdk_versions, content_ratings, others], ys)
        else:
            return (xs, ys)
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.data)




        