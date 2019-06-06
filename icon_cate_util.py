import icon_util
from keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU
from keras.models import Model
from keras_util import group_for_fit_generator
import random
import numpy as np
import math
import preprocess_util
import keras_util

def compute_baseline(aial, aial_test):
    total = 0
    for _,x,_ in aial:
        total += x
    avg = total / len(aial)

    total_mse = 0
    total_mae = 0
    for _,x,_ in aial_test:
        total_mse += (x-avg) ** 2
        total_mae += math.fabs(x-avg)*100/x

    return avg, total_mse/ len(aial_test), total_mae/len(aial_test)

def create_icon_cate_model():
    o = icon_util.create_model(IS_REGRESSION=True)
    input_layer = o['input_layer']
    flatten_layer = o['flatten_layer']
    output_layer = o['output_layer']

    #jump
    x = Dense(32)(flatten_layer)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    output_cate = Dense(18, activation='sigmoid')(x)

    model = Model(input=input_layer, output=[output_layer, output_cate])
    model.compile(optimizer='adam',
        loss={'my_model_regress_1':'mse','dense_2':'binary_crossentropy'},
        metrics={'my_model_regress_1':'mean_absolute_percentage_error'})
    model.summary()
    return model

def datagenerator(aial, batch_size, epochs):

    for i in range(epochs):
        random.shuffle(aial)
        for g in group_for_fit_generator(aial, batch_size, shuffle=True):
            icons = []
            labels = []
            cate_labels = []
            #prepare chrunk
            for app_id, label, cate_label in g:
                try:
                    icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
                    icons.append(icon)
                    labels.append(label)
                    cate_labels.append(cate_label)
                except:
                    pass

            icons = np.asarray(icons)

            #normalize
            icons = icons.astype('float32')
            icons /= 255
            labels = np.array(labels)
            cate_labels = np.array(cate_labels)
            yield icons, [labels, cate_labels]

class FoldData:
    def __init__(self, onehot, avg_rating, std_rating, scamount, total_app):
        self.onehot = onehot
        self.avg_rating = avg_rating
        self.std_rating = std_rating
        self.scamount = scamount
        self.total_app = total_app
    def show(self):
        print(self.onehot, self.avg_rating, self.std_rating, self.scamount, self.total_app)
def fn():
    def makeFoldData(aial):
        onehots = [0] * 18
        total_scamount = 0
        for app_id,rating,onehot, scamount in aial:
            total_scamount += scamount
            for i in range(len(onehot)):
                if onehot[i] == 1:
                    onehots[i]+=1
        avg , std = avg_rating(aial)
        return FoldData(onehots, avg, std, total_scamount, len(aial))
    
    def computeObjValue(fds):
        #onehot
        total_onehot_loss = 0
        for i in range(18):
            maxv = max([fd.onehot[i] for fd in fds])
            minv = min([fd.onehot[i] for fd in fds])
            total_onehot_loss += maxv - minv
        #screenshot
        maxv = max(fd.scamount for fd in fds)
        minv = min(fd.scamount for fd in fds)
        scamount_loss = maxv - minv
        #avg rating
        maxv = max(fd.avg_rating for fd in fds)
        minv = min(fd.avg_rating for fd in fds)
        avg_rating_loss = maxv - minv 
        #std rating
        maxv = max(fd.std_rating for fd in fds)
        minv = min(fd.std_rating for fd in fds)
        std_rating_loss = maxv - minv
        # print(total_onehot_loss, avg_rating_loss, std_rating_loss, scamount_loss)
        return total_onehot_loss * 0.001 + avg_rating_loss * 10 + std_rating_loss * 10 + scamount_loss * 0.0001

    def avg_rating(aial):
        a = np.array([rating for _,rating,_,_ in aial])
        return a.mean() , a.std()

    import random
    answer_list = []
    MAX = 10
    for seed_value in range(476,477):
        print('seed',seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        #prepare data
        aial = preprocess_util.prep_rating_category_scamount()
        random.shuffle(aial)
        fds = []
        for i in range(4):
            train, test = keras_util.gen_k_fold_pass(aial, i, 4)
            fd = makeFoldData(test)
            fds.append(fd)
        #optimize
        loss = computeObjValue(fds)
        if len(answer_list) == 0: answer_list.append((fds,seed_value, loss))
        elif len(answer_list) == MAX: #full pop if better
            if answer_list[-1][2] > loss:
                answer_list.pop()
                answer_list.append((fds, seed_value, loss))
                answer_list = sorted(answer_list, key=lambda x: x[2])
                print(seed_value, loss)
        else:
            answer_list.append((fds, seed_value, loss))
            answer_list = sorted(answer_list, key=lambda x: x[2])
            print(seed_value, loss)
    print(answer_list)
    for fds,seed,loss in answer_list:
        for x in fds: x.show()
        print(seed, loss)

if __name__ == '__main__':
    fn()

