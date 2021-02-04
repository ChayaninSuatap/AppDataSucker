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
from sklearn.metrics import confusion_matrix

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
    output_for_overall_feature = []
    output_for_overall_feature_d = {}
    for app_id, label in preoutput:
        if app_id in sc_dict:
            #add each sc of app_id
            for sc_fn in sc_dict[app_id]:
                output.append( (sc_fn, label))
                output_for_overall_feature_d[sc_fn[:-6]] = label
    for k,v in output_for_overall_feature_d.items():
        output_for_overall_feature.append((k,v))
    return output, output_for_overall_feature

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

def softvote_icon_sc():
    dat = prepare_dataset()
    dat_d = {}

    for k,v in dat:
        dat_d[k] = v
    
    for k in range(4):
        icon_preds = global_util.load_pickle('icon_overall_di_preds_k%s.obj' % (k,))
        sc_preds = global_util.load_pickle('sc_overall_di_preds_k%s.obj' % (k,))

        acc = 0
        count_sample = 0

        y_true = []
        y_pred = []
        pred_d = {}

        for app_id, icon_pred in icon_preds.items():
            total = np.array(icon_pred)
            count = 1
            for i in range(21):
                sc_fn = '%s%2d.png' % (app_id, i)
                if sc_fn in sc_preds:
                    total += sc_preds[sc_fn]
                    count += 1
            averaged = total / count

            pred_d[app_id] = averaged

            if averaged[0] < averaged[1]:
                voted = np.array([0,1])
            else:
                voted = np.array([1,0])
            
            if all(dat_d[app_id] == voted):
                acc += 1
            count_sample += 1
            y_true.append(dat_d[app_id][1])
            y_pred.append(voted[1])
        
        confmat = confusion_matrix(y_true, y_pred, labels = [0, 1])
        print(acc/count_sample)
        print(confmat[0][0], confmat[0][1])
        print(confmat[1][0], confmat[1][1])

        global_util.save_pickle(pred_d, 'softvoted_icon_sc_di_k%d.obj' % (k,))

from overall_feature_util import _all_game_category
def confmat_by_cate(obj_fn):
    dat = prepare_dataset()
    dat_d = {}
    for k,v in dat:
        dat_d[k] = v

    aial = global_util.load_pickle('aial_seed_327.obj')
    cate_d = {}
    for app_id,_, cate_onehot, *_ in aial:
        cate_d[app_id] = _all_game_category[cate_onehot.index(1)]

    o = global_util.load_pickle(obj_fn)

    y_true = {}
    y_pred = {}

    for cate in _all_game_category:
        y_true[cate] = []
        y_pred[cate] = []

    for app_id, pred in o.items():
        if pred[0] < pred[1]:
            normed_pred = np.array([0,1])
        else:
            normed_pred = np.array([1,0])
        
        y_true[cate_d[app_id]].append(dat_d[app_id][1])
        y_pred[cate_d[app_id]].append(normed_pred[1])
        
    confmat_d = {}
    for cate in _all_game_category:
        confmat = confusion_matrix(y_true[cate], y_pred[cate], labels = [0, 1])
        # print(cate)
        # print(confmat[0][0], confmat[0][1])
        # print(confmat[1][0], confmat[1][1])
        confmat_d[cate] = confmat

    return confmat_d

def average_confmat_4folds(fns=[
    'icon_overall_di_preds_k0.obj',
    'icon_overall_di_preds_k1.obj',
    'icon_overall_di_preds_k2.obj',
    'icon_overall_di_preds_k3.obj'
]):
    a = confmat_by_cate(fns[0])
    b = confmat_by_cate(fns[1])
    c = confmat_by_cate(fns[2])
    d = confmat_by_cate(fns[3])
    result_d = {}
    for cate in _all_game_category:
        result_d[cate] = a[cate] + b[cate] + c[cate] + d[cate]
        result_d[cate] = result_d[cate] / 4
    return result_d

def pprint_confmat_d(confmat_d):
    for cate in _all_game_category:
        print(cate)
        print(confmat_d[cate][0][0], confmat_d[cate][0][1])
        print(confmat_d[cate][1][0], confmat_d[cate][1][1])
        prec = confmat_d[cate][1][1]/(confmat_d[cate][0][1] + confmat_d[cate][1][1])
        recall = confmat_d[cate][1][1]/(confmat_d[cate][1][1] + confmat_d[cate][1][0])
        f1 = (prec*recall*2)/(prec+recall)
        print('%.3f' % (f1,))

def make_confmat(test_labels, preds):
    #create y_true
    y_true = []
    for label in test_labels:
        y_true.append(label[1])

    #create y_pred
    y_pred = []
    for pred in preds:
        y_pred.append(0 if pred[0] > pred[1] else 1)

    confmat = confusion_matrix(y_true, y_pred, labels=[0,1])
    return confmat

if __name__ == '__main__':
    dat = prepare_dataset()
    print(dat[:5])

    # softvote_icon_sc()
    # pprint_confmat_d(average_confmat_4folds([
    #     'best_overall_di_k0.obj',
    #     'best_overall_di_k1.obj',
    #     'best_overall_di_k2.obj',
    #     'best_overall_di_k3.obj'
    # ]))

    # pprint_confmat_d(average_confmat_4folds([
    #     'softvoted_icon_sc_di_k1.obj',
    #     'softvoted_icon_sc_di_k2.obj',
    #     'softvoted_icon_sc_di_k3.obj',
    #     'softvoted_icon_sc_di_k0.obj'
    # ]))

    # aial = global_util.load_pickle('aial_seed_327.obj')
    # app_ids_d = {x[0]:True for x in aial}
    # data = _prepare_dataset_sc(
    #     'screenshots.256.distincted.rem.human',
    #     app_ids_d=app_ids_d,
    #     old_db_path='crawl_data/first_version/data.db',
    #     new_db_path='crawl_data/update_first_version_2020_09_12/data.db')

    # print(data[:10])
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



