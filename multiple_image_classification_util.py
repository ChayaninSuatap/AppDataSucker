import random
import preprocess_util
import numpy as np
import icon_cate_util
import keras_util
import sc_util
import mypath
import global_util
import ensemble_model_util

def get_sc_fns_from_fold(k_iter):
    random.seed(859)
    np.random.seed(859)
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    random.shuffle(aial)
    print('aial loss',icon_cate_util.compute_aial_loss(aial))
    icon_cate_util.check_aial_error(aial)

    #filter only rating cate
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    mypath.screenshot_folder = 'screenshots.256.distincted/'
    sc_dict = sc_util.make_sc_dict()

    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k_iter, n_splits=4)
    print(icon_cate_util.compute_baseline(aial_train, aial_test))

    #make aial_train_sc, aial_test_sc
    aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

    #make generator
    gen_test=icon_cate_util.datagenerator(aial_test_sc,
            32, 1, cate_only=True, train_sc=True, shuffle=False, enable_cache=False, yield_app_id=True, skip_reading_image=True)    
    for x in gen_test: yield x #

if __name__ == '__main__':
    #presetting
    k_iter = 3
    # model_fns = ['s1_k3','s2_k3','s3_k3','s4_k3','s5_k3','s6_k3',]
    model_fns_icon = ['i2_k3', 'i3_k3', 'i5_k3', 'i7_k3', 'i10_k3']
    model_fns = ['s1_k3','s2_k3','s3_k3','s4_k3','s5_k3','s6_k3',]
    icon_norm_factor = 1
    sc_norm_factor = 5/6

    sc_fns_k0 = []
    sc_fns_labels_k0 = []
    for x in get_sc_fns_from_fold(k_iter):
        sc_fns_k0 += x[0]
        sc_fns_labels_k0 += [ensemble_model_util.argmax(onehot) for onehot in x[2]]

    #make sc_app_id_labels
    sc_app_ids_labels = {}
    for sc_fn, sc_fn_label in zip(sc_fns_k0, sc_fns_labels_k0):
        app_id = sc_fn[:-6]
        if app_id not in sc_app_ids_labels:
            sc_app_ids_labels[app_id] = sc_fn_label

    #sum predict with listed model
    sum_pred = ensemble_model_util.compute_sum_predict(model_fns)

    #max (sum predict) of a sc_fn and save confident in dict
    max_sum_pred = ensemble_model_util.compute_max_sum(sum_pred, get_confidence=True)
    sc_preds_and_confs_by_fns = {}
    for i,(pred_class, conf) in enumerate(max_sum_pred):
        sc_preds_and_confs_by_fns[sc_fns_k0[i]] = (pred_class, conf)

    #sum predict for app_id from each sc_fn
    dict_app_ids_sum_pred = {}
    for sc_fn in sc_fns_k0:
        app_id = sc_fn[:-6]
        if app_id not in dict_app_ids_sum_pred:
            dict_app_ids_sum_pred[app_id] = [0] * 17
        (pred_class, conf) = sc_preds_and_confs_by_fns[sc_fn]
        dict_app_ids_sum_pred[app_id][pred_class] += conf * sc_norm_factor

    #plus confidence of icon
    dict_app_ids_predclasses_confs_icon = {}
    icon_labels = ensemble_model_util.get_labels(k_iter,True)
    sum_pred_icon = ensemble_model_util.compute_sum_predict(model_fns_icon)
    max_sum_pred_icon = ensemble_model_util.compute_max_sum(sum_pred_icon, get_confidence=True)

    for (app_id, _), (pred_class, conf) in zip(icon_labels, max_sum_pred_icon):
        dict_app_ids_predclasses_confs_icon[app_id] = (pred_class, conf)

    noicon_n=1
    for app_id,(predclass, conf) in dict_app_ids_predclasses_confs_icon.items():
        if app_id in dict_app_ids_sum_pred:
            dict_app_ids_sum_pred[app_id][predclass] += conf * icon_norm_factor
        else:
            print('no icon', noicon_n)
            noicon_n += 1
    
    #topk accuracy
    for topk in range(1,6):
        acc = 0
        for k in dict_app_ids_sum_pred.keys():
            argsorted = np.flip(np.argsort(dict_app_ids_sum_pred[k]))[:topk]
            if sc_app_ids_labels[k] in argsorted:
                acc += 1
        print(acc/len(dict_app_ids_sum_pred), end=' ')
    print('\n')

    # #max sum of predict
    # dict_app_ids_pred_class = {}
    # for k in dict_app_ids_sum_pred.keys():
    #     dict_app_ids_pred_class[k] = ensemble_model_util.argmax( dict_app_ids_sum_pred[k])

    # #measure accuracy
    # acc = 0
    # for k,item in dict_app_ids_sum_pred.items():
    #     if dict_app_ids_pred_class[k] == sc_app_ids_labels[k]:
    #         acc += 1
    # print(acc / len(dict_app_ids_pred_class), len(dict_app_ids_pred_class))


    
        
        
