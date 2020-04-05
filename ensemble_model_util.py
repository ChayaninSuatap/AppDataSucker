from global_util import load_pickle, save_pickle
import random
import numpy as np
import preprocess_util
import icon_cate_util
from keras_util import gen_k_fold_pass
import icon_util
import sc_util
import os
import mypath
import keras_util

def argmax(l):
    return np.array(l).argmax()

def compute_sum_predict(model_fns, file_extension='obj', directory='ensemble_data'):
    output = None
    for i,model_fn in enumerate(model_fns):
        o = load_pickle(directory + '/' + model_fn + '.' + file_extension)
        if i == 0: output = o
        else:
            output += o
    return output

def make_ensemble_data_human_testset(ensemble_data, labels, human_testset_labels):
    output = []
    for app_id_human,_ in human_testset_labels:

        for i, (app_id_labels,_) in enumerate(labels):
            if app_id_human == app_id_labels:
                output.append( ensemble_data[i])
                break
    
    return output

def compute_max_sum(sum_pred, get_confidence=False):
    output = []
    for pred_row in sum_pred:
        if get_confidence==False:
            output.append(argmax(pred_row))
        else:
            output.append((argmax(pred_row), max(pred_row)))
    return output

def get_labels(k_fold, get_app_id=False):
    random.seed(859)
    np.random.seed(859)
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    random.shuffle(aial)
    print('aial loss',icon_cate_util.compute_aial_loss(aial))
    icon_cate_util.check_aial_error(aial)

    #filter only rating cate
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)

    aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=k_fold, n_splits=4)

    output=[]
    for app_id,_,cate in aial_test:
        try:
            icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
        except:
            continue
        if get_app_id:
            output.append(( app_id, np.array(cate).argmax()))
        else:
            output.append( np.array(cate).argmax())
    return output

def get_labels_t(k, aial_path = 'aial_seed_327.obj'):
    mypath.icon_folder = 'similarity_search/icons_rem_dup_human_recrawl/'
    random.seed(327)
    np.random.seed(327)
    aial = icon_cate_util.make_aial_from_seed(327, 'similarity_search/icons_rem_dup_human_recrawl/')
    aial = icon_cate_util.filter_aial_rating_cate(aial)
    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k, n_splits=4)

    output = []
    for app_id,_,cate in aial_test:
        output.append(  np.array(cate).argmax())
    return output

def get_labels_sc(seed_value, sc_dir, k_fold, aial_path = 'aial_seed_327.obj'):
    import mypath
    mypath.screenshot_folder = sc_dir
    sc_dict = sc_util.make_sc_dict(sc_dir)
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    aial = load_pickle(aial_path)
    print('aial loss',icon_cate_util.compute_aial_loss(aial))
    icon_cate_util.check_aial_error(aial)
    
    #filter only rating cate
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=k_fold, n_splits=4)
    
    #make aial_train_sc, aial_test_sc
    _, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

    output=[]
    for app_id,_,cate in aial_test_sc:
        output.append( np.array(cate).argmax())
    return output

def get_human_testset_labels():
    o=load_pickle('app_ids_for_human_test.obj')
    return remove_app_id(o)

def remove_app_id(labels):
    return [x for _,x in labels]

def compute_topk_accuracy(pred, labels, topk):
    correct_count = 0
    for p, label in zip(pred, labels):
        argsorted = np.flip(np.argsort(p))[:topk]
        if label in argsorted:
            correct_count += 1
    return correct_count/len(pred)

def compute_predict_labels(pred):
    return [argmax(x) for x in pred]

def check_len_preds_len_labels(labels, ensemble_fd, k_iter):
    labels_n = len(labels)
    print('len(labels)', labels_n)

    for fn in os.listdir(ensemble_fd):
            if 'k' + str(k_iter) in fn:
                print(fn)
                o = load_pickle(ensemble_fd+ fn)
                print(len(o))
                if len(o) != labels_n:
                    raise Exception('len preds and labels error')
    
    print('len preds and labels ok')

def get_pred_fns_by_k(ensemble_fd, k_iter):

    return [fn for fn in os.listdir(ensemble_fd) if 'k' + str(k_iter) in fn]

def filter_pred_fns(ensemble_fd, filter_fn):
    return [fn for fn in os.listdir(ensemble_fd) if filter_fn(fn)]

def create_icon_human_set_obj(icon_fd):
    xs = []
    for fn in sorted(os.listdir(icon_fd), key = lambda x: int(x[:-4])):
        print(fn)
        icon = icon_util.load_icon_by_fn(icon_fd + fn, 128, 128)
        icon = icon.astype('float32') / 255
        xs.append(icon)
    save_pickle( (np.array(xs), np.array(get_human_testset_labels())), 'icon_human_testset.obj') 

def create_sc_human_set_obj(sc_fd):
    xs = []
    for fn in sorted(os.listdir(sc_fd), key = lambda x: int(x[:-4])):
        print(fn)
        sc = icon_util.load_icon_by_fn(sc_fd + fn, 256, 160, rotate_for_sc = True)
        sc = sc.astype('float32') / 255
        xs.append(sc)
    save_pickle( (np.array(xs), np.array(get_human_testset_labels())), 'sc_human_testset.obj') 

def create_all_sc_human_set_obj(sc_fd ,human_app_ids_path = 'app_ids_for_human_test.obj'):
    sc_dict = sc_util.make_sc_dict(sc_fd)
    o = load_pickle(human_app_ids_path)
    xs = []
    labels = []
    count_app_id_error = 0
    for app_id, label in o:
        labels.append(label)

        if app_id in sc_dict:
            scs = []
            for sc_fn in sc_dict[app_id]:
                try:
                    sc = icon_util.load_icon_by_fn(sc_fd + sc_fn, 256, 160, rotate_for_sc = True)
                    sc = sc.astype('float32') / 255
                    scs.append(sc)
                    print(sc_fn)
                except:
                    pass
            xs.append(scs)
            print('len(scs)', len(scs))
        else:
            count_app_id_error += 1
        
    save_pickle( (xs, labels), 'all_sc_human_testset.obj')
    print('error', count_app_id_error)

def compute_topk_all_sc_of_game(sc_dir, k_fold,
    ensemble_sc_fd, sc_model_fns,
    ensemble_icon_fd=None, icon_model_fns=None,
    aial_path = 'aial_seed_327.obj'):
    import mypath
    mypath.screenshot_folder = sc_dir
    sc_dict = sc_util.make_sc_dict(sc_dir)
    aial = load_pickle(aial_path)
    print('aial loss',icon_cate_util.compute_aial_loss(aial))
    icon_cate_util.check_aial_error(aial)
    
    #filter only rating cate
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
    aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=k_fold, n_splits=4)
    
    #make aial_train_sc, aial_test_sc
    _, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

    #make aial_test_app_id
    aial_test_app_id = [x[0] for x in aial_test]
    labels = []
    #make index_dict
    index_dict = {}
    for i, (sc_fn, _, _) in enumerate(aial_test_sc):
        index_dict[sc_fn] = i
    
    #load model
    models = [load_pickle(ensemble_sc_fd + model) for model in sc_model_fns]
    if ensemble_icon_fd != None:
        icon_models = [load_pickle(ensemble_icon_fd + model) for model in icon_model_fns]

    dataset_pred = []

    #pred scs for each game
    for aial_test_i, (app_id, _, label) in enumerate(aial_test):

        game_pred = []

        if app_id in sc_dict:
            for sc_fn in sc_dict[app_id]:

                #screenshot
                sc_fn_pred = models[0][index_dict[sc_fn]]

                for model in models[1:]:
                    sc_fn_pred += model[index_dict[sc_fn]]
                
                #normalize sc case
                for i in range(len(sc_fn_pred)): sc_fn_pred[i] /= len(models) 

                if len(game_pred) == 0:
                    game_pred = sc_fn_pred
                else:
                    for i in range(17): game_pred[i] += sc_fn_pred[i]

            if ensemble_icon_fd != None: 
                #icon
                icon_fn_pred = icon_models[0][aial_test_i]
                for model in icon_models[1:]:
                    icon_fn_pred += model[aial_test_i]
                
                #normalize icon case
                for i in range(len(icon_fn_pred)): icon_fn_pred[i] /= len(icon_models)
                #add icon eval to preds
                for i in range(17): game_pred[i] += icon_fn_pred[i]

            dataset_pred.append(game_pred)
            labels.append(np.argmax(label))
    
    print(len(dataset_pred), len(labels))
    for i in range(1,6):
        print(compute_topk_accuracy(dataset_pred, labels, i), end=" ")
        

if __name__ == '__main__':

    k = 3
    
    # model_prefixes = [
    #    'sc_model2.3_k%d_no_aug' % (k,),
    #    'sc_model2.4_k%d_no_aug' % (k,),
    #    'sc_model1.6_k%d_no_aug' % (k,),
    #    'sc_model1.5_k%d_no_aug' % (k,),
    #    'sc_model1.3_k%d_no_aug' % (k,),
    # ]

    model_prefixes = [
       'sc_model1.1_k%d_no_aug' % (k,),
       'sc_model1.2_k%d_no_aug' % (k,),
       'sc_model1.3_k%d_no_aug' % (k,),
       'sc_model1.4_k%d_no_aug' % (k,),
       'sc_model1.5_k%d_no_aug' % (k,),
       'sc_model1.6_k%d_no_aug' % (k,),
    ]

    icon_model_prefixes = [
        'icon_model2.4_k%d_t' % (k,),
        'icon_model2.3_k%d_t' % (k,),
        'icon_model1.3_k%d_t' % (k,),
        'icon_model1.2_k%d_t' % (k,),
        'icon_model1.5_k%d_t' % (k,),
    ]

    # icon_model_prefixes = [
    #     'icon_model1.1_k%d_t' % (k,),
    #     'icon_model1.2_k%d_t' % (k,),
    #     'icon_model1.3_k%d_t' % (k,),
    #     'icon_model1.4_k%d_t' % (k,),
    #     'icon_model1.5_k%d_t' % (k,),
    #     'icon_model1.6_k%d_t' % (k,),
    # ]

    model_fns = []
    for fn in os.listdir('ensemble_model_predicts_t/sc/'):
        if fn[:21] in model_prefixes:
            model_fns.append(fn)
    
    icon_model_fns = []
    for fn in os.listdir('ensemble_model_predicts_t/icon/'):
        if fn[:18] in icon_model_prefixes:
            icon_model_fns.append(fn)

    compute_topk_all_sc_of_game('screenshots.256.distincted.rem.human/', k,
        'ensemble_model_predicts_t/sc/', model_fns,
        'ensemble_model_predicts_t/icon/', icon_model_fns)


    # k_iter = 3
    # ensemble_fd = 'ensemble_model_predicts_t/icon/'
    # labels = get_labels_t(k_iter)

    # check_len_preds_len_labels(labels, ensemble_fd, k_iter)
    # # labels = get_labels_sc(327, 'screenshots.256.distincted.rem.human/', 3)

    # def filter_fn(fn, k_iter):
    #     valid_subnames = ['2.3', '2.4', '1.2', '1.5', '1.3']
    #     # print(list(map(lambda x: 'sc_model' + x, valid_subnames)))
    #     return 'k' + str(k_iter) in fn and fn[:13] in list(map(lambda x: 'icon_model' + x, valid_subnames))
    #     # return 'k' + str(k_iter) in fn and fn[:11] == 'icon_model1'
    
    # model_fns = list(map(lambda x: x[:-4], filter_pred_fns(ensemble_fd, lambda fn: filter_fn(fn, k_iter))))
    # print(model_fns)

    # sum_pred = compute_sum_predict(model_fns, directory = ensemble_fd[:-1])    

    # for i in range(1,6):
    #     print(compute_topk_accuracy(sum_pred, labels, i), end=" ")

    # t stuff
    # model_fns = map(lambda x: x[:-4], os.listdir('ensemble_model_predicts_t'))

    # model_fns = ['icon_model1.2_k0_t_human', 'icon_model1.3_k0_t_human', 'icon_model1.5_k0_t_human', 'icon_model1.1_k0_t_human', 'icon_model1.4_k0_t_human']

    # sum_pred = compute_sum_predict(model_fns, directory='ensemble_model_predicts_t')

    # labels = get_labels_t(3)
    # labels = remove_app_id(get_human_testset_labels())

    # for i in range(1,6):
    #     print(compute_topk_accuracy(sum_pred, labels, i), end=" ")
    

    #paper stuff
    # labels = get_labels(0, get_app_id=False)
    # labels = get_labels_sc(2)

    # save_pickle(labels, 'ensemble_model_icon_labels_k0')
    # o = load_pickle('ensemble_data/s1_k0.obj')
    # input(o)

    # sum_pred = compute_sum_predict(['i1_k3', 'i2_k3', 'i3_k3', 'i4_k3', 'i5_k3', 'i6_k3',])
    # sum_pred = compute_sum_predict(['i2_k0', 'i3_k0', 'i5_k0', 'i7_k0', 'i10_k0',])

    # sum_pred = compute_sum_predict(['s1_k0_h', 's2_k0_h', 's3_k0_h', 's4_k0_h', 's5_k0_h', 's6_k0_h',])
    # sum_pred = compute_sum_predict(['s3_k0_h', 's5_k0_h', 's6_k0_h', 's7_k0_h', 's10_k0_h',]) 
    # labels = remove_app_id(get_human_testset_labels())
    # sum_pred = make_ensemble_data_human_testset(sum_pred, labels, human_testset_labels)

    #check dataset match
    # print(len(sum_pred), len(labels))
    
    #compute top5
    # for i in range(1,6):
        # print(compute_topk_accuracy(sum_pred, labels, i), end=" ")

    # sum_pred = compute_sum_predict(['i1_k0'])
    # max_sum = compute_max_sum(sum_pred)
    # print(compute_accuracy(max_sum, labels))
    # print(compute_topk_accuracy(sum_pred, labels, 1), end=" ")

    # print(compute_topk_accuracy(sum_pred, remove_app_id(human_testset_labels), 1))
    # [print(x) for x in compute_predict_labels(sum_pred)]

