from global_util import load_pickle, save_pickle
import random
import numpy as np
import preprocess_util
import icon_cate_util
from keras_util import gen_k_fold_pass
import icon_util
import sc_util


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

def get_labels_sc(k_fold):
    import mypath
    mypath.screenshot_folder = 'screenshots.256.distincted/'
    sc_dict = sc_util.make_sc_dict()
    
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
    
    #make aial_train_sc, aial_test_sc
    _, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

    output=[]
    for app_id,_,cate in aial_test_sc:
        # try:
        #     icon = icon_util.load_icon_by_fn(mypath.screenshot_folder + app_id, 256, 160, rotate_for_sc=True)
        # except Exception as e:
        #     print(repr(e))
        #     continue
        output.append( np.array(cate).argmax())
    return output

def get_human_testset_labels():
    o=load_pickle('app_ids_for_human_test.obj')
    return o

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

if __name__ == '__main__':
    # labels = get_labels(0, get_app_id=False)
    # labels = get_labels_sc(2)

    # save_pickle(labels, 'ensemble_model_icon_labels_k0')
    # o = load_pickle('ensemble_data/s1_k0.obj')
    # input(o)

    # sum_pred = compute_sum_predict(['i1_k3', 'i2_k3', 'i3_k3', 'i4_k3', 'i5_k3', 'i6_k3',])
    # sum_pred = compute_sum_predict(['i2_k0', 'i3_k0', 'i5_k0', 'i7_k0', 'i10_k0',])

    # sum_pred = compute_sum_predict(['s1_k0_h', 's2_k0_h', 's3_k0_h', 's4_k0_h', 's5_k0_h', 's6_k0_h',])
    sum_pred = compute_sum_predict(['s3_k0_h', 's5_k0_h', 's6_k0_h', 's7_k0_h', 's10_k0_h',]) 
    labels = remove_app_id(get_human_testset_labels())
    # sum_pred = make_ensemble_data_human_testset(sum_pred, labels, human_testset_labels)

    #check dataset match
    print(len(sum_pred), len(labels))
    
    #compute top5
    # for i in range(1,6):
        # print(compute_topk_accuracy(sum_pred, labels, i), end=" ")

    # sum_pred = compute_sum_predict(['i1_k0'])
    # max_sum = compute_max_sum(sum_pred)
    # print(compute_accuracy(max_sum, labels))
    # print(compute_topk_accuracy(sum_pred, labels, 1), end=" ")

    # print(compute_topk_accuracy(sum_pred, remove_app_id(human_testset_labels), 1))
    [print(x) for x in compute_predict_labels(sum_pred)]

