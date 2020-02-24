import icon_util
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, MaxPooling2D, Flatten, Dropout, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.models import Model, load_model
from keras_util import group_for_fit_generator
import random
import numpy as np
import math
import preprocess_util
import keras_util
import matplotlib.pyplot as plt
import mypath
import os
from datetime import datetime
import global_util

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

def create_icon_cate_model(cate_only=False, is_softmax=False, use_gap=False, train_sc=False, layers_filters = [64, 128, 256], dropout=0.2,
    sliding_dropout=None , conv1x1_layer_n=1, stack_conv=1, do_slide_down=False, conv1x1_reduce_rate=2, predict_rating=False,
    disable_conv_dropout=False, layers_dense=[34], conv1x1_maxpool=True):

    dropout_for_conv = 0 if disable_conv_dropout else dropout
    sliding_dropout_for_conv = None if disable_conv_dropout else sliding_dropout

    o = icon_util.create_model(IS_REGRESSION=True, use_gap=use_gap, train_sc=train_sc, layers_filters=layers_filters, dropout=dropout_for_conv,
        sliding_dropout=sliding_dropout_for_conv, conv1x1_layer_n=conv1x1_layer_n, stack_conv=stack_conv, do_slide_down=do_slide_down,
        conv1x1_reduce_rate=conv1x1_reduce_rate, conv1x1_maxpool=conv1x1_maxpool)

    input_layer = o['input_layer']
    flatten_layer = o['flatten_layer']
    output_layer = o['output_layer']

    #assigning output
    if use_gap and is_softmax:
        output_cate = Dense(17, activation='softmax')(flatten_layer)
    else:
        #first dense
        x = Dense(layers_dense[0])(flatten_layer)
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)

        #regular dropout
        if sliding_dropout==None:
            x = Dropout(dropout, name='do_dense_0_%.2f' % (dropout,))(x)
        #increasing dropout value
        else:
            current_dropout_value = o['current_dropout_value'] if not disable_conv_dropout else sliding_dropout[0]
            x = Dropout(current_dropout_value, name='do_dense_0_'+ str(current_dropout_value))(x)
            current_dropout_value += sliding_dropout[1]

        #later dense (before softmax layer)
        for i in range(1, len(layers_dense)):
            x = Dense(layers_dense[i])(x)
            x = LeakyReLU()(x)
            x = BatchNormalization()(x)
            #regular dropout
            if sliding_dropout==None:
                x = Dropout(dropout, name='do_dense_%d_%.2f' % (i,dropout,))(x)
            #increasing dropout value
            else:
                x = Dropout(current_dropout_value, name= 'do_dense_%d_%f' % (i,current_dropout_value))(x)
                current_dropout_value += sliding_dropout[1]


        #predict class
        output_cate = Dense(17, activation='softmax')(x)
        if predict_rating:
            output_cate = Dense(1, activation='linear')(output_cate)
    
    model_output = output_cate if cate_only else [output_layer, output_cate]
    model = Model(inputs=input_layer, outputs=model_output)

    #compilation
    if cate_only and not predict_rating:
        model.compile(optimizer='adam',
            loss='categorical_crossentropy' if is_softmax else 'binary_crossentropy',
            metrics=['acc'])
    elif cate_only and predict_rating:
        model.compile(optimizer='adam',
            loss='mse',
            metrics=['mape'])
    model.summary()
    return model

def datagenerator(aial, batch_size, epochs, cate_only=False, train_sc=False, shuffle=True, enable_cache=False, limit_cache_n=None,
    yield_app_id=False, skip_reading_image=False, predict_rating=False, icon_resize_dim=(128, 128), skip_reading_amount=0, datagen=None):

    cache_dict = {}
    
    #limit cache 
    if limit_cache_n != None: cached_n = 0

    for i in range(epochs):

        for g in group_for_fit_generator(aial, batch_size, shuffle=shuffle):
            icons = []
            labels = []
            cate_labels = []
            app_ids = []
            #prepare chrunk
            for app_id, label, cate_label in g:
                #get from cache
                if enable_cache and app_id in cache_dict:
                    icon = cache_dict[app_id]
                else:
                #read img from dir
                    try:
                        if skip_reading_image:
                            continue
                        elif skip_reading_amount > 0:
                            skip_reading_amount -= 1
                            continue
                        elif train_sc:
                            icon = icon_util.load_icon_by_fn(mypath.screenshot_folder + app_id, 256, 160, rotate_for_sc=True)
                        elif not train_sc:
                            icon = icon_util.load_icon_by_app_id(app_id, icon_resize_dim[0], icon_resize_dim[0])
                    except:
                        continue
                    #put in cache
                    if enable_cache:
                    
                        #limit cache
                        if limit_cache_n != None:
                            #cache only when below limit
                            if  cached_n < limit_cache_n:
                                cache_dict[app_id] = icon
                                cached_n += 1
                        # no limit cache
                        else:
                            cache_dict[app_id] = icon
                app_ids.append(app_id)
                icons.append(icon)
                labels.append(label)
                cate_labels.append(cate_label)     

            icons = np.asarray(icons)

            #test dataget
            # old_icons = np.array(icons)

            if datagen:
                for augmented_chrunk in datagen.flow(icons, batch_size = icons.shape[0], shuffle=False):
                    icons_temp = augmented_chrunk
                    break
                icons = icons_temp
            
            #test datagen
            # fig, axes = plt.subplots(2, 2)
            # for old_icon, icon in zip(old_icons, icons):
            #     axes[0,0].imshow(old_icon.astype('float32') / 255)
            #     axes[0,1].imshow(icon.astype('float32') / 255)
            #     plt.show()

            #normalize
            icons = icons.astype('float32')
            icons /= 255
            labels = np.asarray(labels)
            cate_labels = np.asarray(cate_labels)

            #yield
            if cate_only and not predict_rating:
                yield (icons, cate_labels) if not yield_app_id else (app_ids, icons, cate_labels)
            elif cate_only and predict_rating:
                yield (icons, labels) if not yield_app_id else (app_ids, icons, labels)
            else:
                yield (icons, [labels, cate_labels]) if not yield_app_id else (app_ids, icons, [labels, cate_labels])
                
class FoldData:
    def __init__(self, onehot, avg_rating, std_rating, scamount, total_app, download_dict, rating_amount_dict):
        self.onehot = onehot
        self.avg_rating = avg_rating
        self.std_rating = std_rating
        self.scamount = scamount
        self.total_app = total_app
        self.download_dict = download_dict
        self.rating_amount_dict = rating_amount_dict
    def show(self):
        print(self.onehot, self.avg_rating, self.std_rating, self.scamount, self.total_app, self.download_dict, self.rating_amount_dict)
def _makeFoldData(aial):
    onehots = [0] * 17
    total_scamount = 0
    download_dict = [0,0,0,0]
    rating_amount_dict = [0,0,0,0]
    for app_id,rating,onehot, scamount, download, rating_amount in aial:
        total_scamount += scamount
        for i in range(len(onehot)):
            if onehot[i] == 1:
                onehots[i]+=1
        #download
        if download >= 1_000_000: download_dict[3] +=1
        elif download >= 1_00_000: download_dict[2] +=1
        elif download >= 5_000: download_dict[1] +=1
        else: download_dict[0] += 1
        #rating amount
        if rating_amount >= 5000: rating_amount_dict[3] += 1
        elif rating_amount >= 500: rating_amount_dict[2] += 1
        elif rating_amount >= 100: rating_amount_dict[1] += 1
        else: rating_amount_dict[0] += 1
    avg , std = _avg_rating(aial)
    return FoldData(onehots, avg, std, total_scamount, len(aial), download_dict, rating_amount_dict)

def _computeObjValue(fds):
    #onehot
    total_onehot_loss = 0
    for i in range(17):
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

def _avg_rating(aial):
    a = np.array([x[1] for x in aial])
    return a.mean() , a.std()
def _make_fds(aial):
    fds = []
    for i in range(4):
        train, test = keras_util.gen_k_fold_pass(aial, i, 4)
        fd = _makeFoldData(test)
        fds.append(fd)
    return fds
def compute_aial_loss(aial):
    return _computeObjValue(_make_fds(aial))

def make_aial_from_seed(seed, icon_fd):
    #prepare data
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    #use only app_id in this folder
    new_aial = []
    icon_fns = os.listdir(icon_fd)
    for x in aial:
        if (x[0] + '.png') in icon_fns:
            new_aial.append(x)
    random.seed(seed)
    np.random.seed(seed)
    random.shuffle(new_aial)
    return new_aial

def dump_aial_from_seed(seed, icon_fd, dump_full_path):
    aial = make_aial_from_seed(seed, icon_fd)
    global_util.save_pickle(aial, dump_full_path)
    return aial

def filter_aial_rating_cate(aial):
    newaial = []
    for x in aial:
        newaial.append( (x[0], x[1], x[2]))
    return newaial

def find_best_seed(icon_fd):
    import random
    answer_list = []
    MAX = 10
    #check manually
    # aial = make_aial_from_seed(772, icon_fd)
    # print(compute_aial_loss(aial))
    # train, test = keras_util.gen_k_fold_pass(aial, 0, 4)
    # fd=_makeFoldData(test)
    # fd.show()
    # input()

    #prepare data
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    #use only app_id in this folder
    new_aial = []
    icon_fns = os.listdir(icon_fd)
    for x in aial:
        if (x[0] + '.png') in icon_fns:
            new_aial.append(x)
    aial_master = new_aial

    for seed_value in range(0,1000):
        print('seed',seed_value)
        random.seed(seed_value)
        np.random.seed(seed_value)
        
        aial = []
        for x in aial_master:
            aial.append(x)

        random.shuffle(aial)
        fd=_makeFoldData(aial)
        fds = []
        for i in range(4):
            train, test = keras_util.gen_k_fold_pass(aial, i, 4)
            fd = _makeFoldData(test)
            fds.append(fd)
        #optimize
        loss = _computeObjValue(fds)
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

def check_aial_error(aial):
    fds = _make_fds(aial)
    for x in fds: x.show()
    for x in aial:
        if all(y==0 for y in x[2]): print('all zero')
        if sum(x[2])>1: print('shit')

def eval_top_k(gen_test, steps, model=None):
    import os
    from keras_util import metric_top_k
    import keras
    for fn in os.listdir('eval_top_k'):
        path = 'eval_top_k/' + fn
        #load model or load weights
        if model==None:
            model = load_model(path)
        else:
            model.load_weights(path)
        #compile and eval
        model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['acc',metric_top_k(2),metric_top_k(3),metric_top_k(4), metric_top_k(5)])
        print(path)
        print(model.evaluate_generator(gen_test, steps=steps))
        print("")
    print('done')

def plot_confusion_matrix_generator_icon_cate(model, test_gen_for_ground_truth, test_gen_for_predict, steps_per_epoch):
    ground_truth = []
    for _, labels in test_gen_for_ground_truth:
        for label in labels: ground_truth.append(label)
    keras_util.plot_confusion_matrix_generator(model, ground_truth, test_gen_for_predict, steps_per_epoch)

def compute_class_weight_for_cate(aial_train):
    return keras_util.compute_class_weight([np.argmax(x) for _,_,x in aial_train])

if __name__ == '__main__':
    #seed 772 for icons.rem.duplicate
    #0.523598116128659
    aial = dump_aial_from_seed(327, 'similarity_search/icons_rem_dup_human_recrawl/', 'aial_seed_327.obj')
    print(compute_aial_loss(aial))
    # find_best_seed('similarity_search/icons_remove_duplicate(for icons.backup)')