import db_util
import global_util
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
import numpy as np
import overall_feature_util
from sklearn.utils.class_weight import compute_class_weight as sk_compute_class_weight
import keras_util
import random
import os
from sc_util import make_sc_dict

def _prepare_dataset(app_ids_d, old_db_path, new_db_path):
    old_conn = db_util.connect_db(old_db_path)
    new_conn = db_util.connect_db(new_db_path)

    old_d = {}
    new_d = {}
    for dat in old_conn.execute('select app_id, download_amount from app_data'):
        if dat[0] in app_ids_d:
            old_d[dat[0]] = int(dat[1].replace(',','').replace('+',''))
    
    for dat in new_conn.execute('select app_id, download_amount from app_data'):
        if dat[0] in app_ids_d:
            new_d[dat[0]] = int(dat[1].replace(',','').replace('+',''))
    
    output = []
    for k_old, v_old in old_d.items():
        if k_old in new_d:
            if new_d[k_old] > v_old:
                output.append((k_old, np.array([0,1])))
            else:
                output.append((k_old, np.array([1,0])))
    return output

def _prepare_dataset_sc(sc_fd, **kwargs):
    preoutput = _prepare_dataset(**kwargs)
    sc_dict = make_sc_dict(sc_fd)
    output = []
    for app_id, label in preoutput:
        if app_id in sc_dict:
            #add each sc of app_id
            for sc_fn in sc_dict[app_id]:
                output.append( (sc_fn, label))
    return output

def prepare_dataset():
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    output = _prepare_dataset(
        app_ids_d,
        'crawl_data/first_version/data.db',
        'crawl_data/update_first_version_2020_09_12/data.db')
    return output

def extend_cate_model(model):
    input_layer = model.layers[0].input
    last_layer = model.layers[-2].output
    last_layer = Dense(2, name='class_download_increase', activation='softmax')(last_layer)
    model = Model(inputs=[input_layer], outputs=[last_layer])
    model.compile(optimizer='adam',
            loss='categorical_crossentropy', metrics=['acc'])
    return model

def compute_class_weight(train_labels):
    y_ints = np.argmax(train_labels, axis=1)
    class_weights = sk_compute_class_weight('balanced', np.unique(y_ints), y_ints)
    return (dict(enumerate(class_weights)))
    
if __name__ == '__main__':
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    data = _prepare_dataset_sc(
        'screenshots.256.distincted.rem.human',
        app_ids_d=app_ids_d,
        old_db_path='crawl_data/first_version/data.db',
        new_db_path='crawl_data/update_first_version_2020_09_12/data.db')
    print(data[:10])
    # random.seed(5)
    # random.shuffle(data)
    # train, test = keras_util.gen_k_fold_pass(data, kf_pass=3, n_splits=4)
    # f_train = open('download_increase_for_ajk/train_k3.txt', 'w', encoding='utf-8')
    # f_test = open('download_increase_for_ajk/test_k3.txt', 'w', encoding='utf-8')
    # for x in train:
    #     print('%s.png %d' % (x[0], x[1][1]), file=f_train)
    # for x in test:
    #     print('%s.png %d' % (x[0], x[1][1]), file=f_test)

    # f_train.close()
    # f_test.close()
    # print(len(train),len(test))
    # c0, c1 = 0, 0
    # for x in test:
    #     if x[-1][0] == 1: c0+=1
    #     else: c1+=1
    # print(c0, c1)

    # model = load_model('sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5')
    # model = extend_cate_model(model)
    # model.summary()
    # print(model.layers[-2].name)
    # print(model.layers[-1].name)



