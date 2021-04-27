import os
from preprocess_util import filter_sc_fns_from_icon_fns_by_app_id
from global_util import load_pickle, save_pickle
from joblib import dump, load
import numpy as np
import keras_util
import random
import math
from basic_features import my_fit_scaler, my_transform_to_scaler, make_model, extract_hog, load_dataset
from basic_features_t_augment import split_aial, make_test_set
from tensorflow.keras.models import load_model

def make_not_computed_gist_sc_list(old_sc_gist_txt, sc_fns, app_ids):
    old_computed_list = {}
    for line in open(old_sc_gist_txt):
        splited = line.split(' ')[:-512]

        if len(splited) == 1:
            sc_fn = splited[0]
        elif len(splited) == 2:
            sc_fn = splited[0] + ' ' + splited[1]
        else:
            raise Exception('splited > 2')
        
        old_computed_list[sc_fn] = 1
    
    not_computed_sc_list = []
    found_in_app_ids_count = 0
    for sc_fn in sc_fns:
        if sc_fn[:-6] in app_ids: found_in_app_ids_count += 1
        if sc_fn not in old_computed_list and sc_fn[:-6] in app_ids:
            not_computed_sc_list.append(sc_fn)
    
    print('found in app_ids', found_in_app_ids_count)
    print(not_computed_sc_list)
    print(len(not_computed_sc_list))
    
def check_gist_has_all_icons(gist_path, app_ids):
    f = open(gist_path)
    for line in f:
        splited = line.split(' ')
        app_id = splited[0]
        features = splited[1:]
        if len(features) != 512:
            raise ValueError('%s has only %d shape feature' % (app_id, len(features)))
        if app_id not in app_ids:
            raise ValueError('%s is not found in app_ids' % (app_id,))
    print('dataset is OK')

def make_gist_obj(source, dest):
    f = open(source)
    output = {}
    for line in f:
        splited = line.split(' ')
        first_splited = splited[:-512]
        second_splited = splited[-512:]

        if len(first_splited) == 1:
            sc_fn = first_splited[0]
        elif len(first_splited) == 2:
            sc_fn = first_splited[0] + ' ' + first_splited[1]
        else:
            raise Exception('splited > 2')

        app_id = sc_fn
        features = np.array([float(x) for x in second_splited])
        if len(features) != 512 :
            raise ValueError('feature dim error')
        if not math.isnan(features[0]):
            output[app_id] = features
        else:
            print('found NAN, not added', app_id)
    save_pickle(output, dest)
    
#used in both hog & gist
def split_train_test(dataset_path, train_path, test_path, k_iter, aial_obj, compress=3, sc=False):
    #split train test from dataset and also normalize it
    gist_dict = load(dataset_path)

    #make aial_train, aial_test
    if sc:
        aial_train, aial_test = keras_util.gen_k_fold_pass(aial_obj, kf_pass = k_iter, n_splits=4)
        train_d = {x[0]:x for x in aial_train}
        test_d  = {x[0]:x for x in aial_test}
        aial_train_new = []
        aial_test_new = []
        #make app_id key to app_id & sc number key
        for k in gist_dict.keys():
            app_id = k[:-6]
            if app_id in train_d:
                new_rec = list(train_d[app_id])
                new_rec[0] = k
                aial_train_new.append(tuple(new_rec))
            if app_id in test_d:
                new_rec = list(test_d[app_id])
                new_rec[0] = k
                aial_test_new.append(tuple(new_rec))
        aial_train = aial_train_new
        aial_test = aial_test_new

    else:
        aial_train, aial_test = keras_util.gen_k_fold_pass(aial_obj, kf_pass = k_iter, n_splits=4)

    gist_train_dict = {}
    gist_test_dict = {}

    # aial train & test need cate !
    for app_id,_,_,_,_,_ in aial_train:
        if app_id in gist_dict :
            gist_train_dict[app_id] = gist_dict[app_id]
            del gist_dict[app_id]

    for app_id,_,_,_,_,_ in aial_test:
        if app_id in gist_dict :
            gist_test_dict[app_id] = gist_dict[app_id]
            del gist_dict[app_id]

    #normalize
    mean, var = my_fit_scaler(gist_train_dict)
    print('mean', mean, 'var', var)
    my_transform_to_scaler(gist_train_dict, mean, var)
    my_transform_to_scaler(gist_test_dict, mean, var)

    #add label
    for app_id,_,cate,_,_,_  in aial_train:
        if app_id in gist_train_dict : gist_train_dict[app_id] = gist_train_dict[app_id], cate

    for app_id,_,cate,_,_,_  in aial_test:
        if app_id in gist_test_dict : gist_test_dict[app_id] = gist_test_dict[app_id], cate

    dump(gist_train_dict, train_path, compress=compress)
    del gist_train_dict
    dump(gist_test_dict, test_path, compress=compress)
    del gist_test_dict
    print('done')

#one time extract
def extract_hog16_sc_from_file(source, dest, aial_dest):
    if not os.path.exists(dest):
        os.mkdir(dest)
    
    o = load(source)
    aial_d = {}
    for app_id, (x,y) in o.items():
        save_pickle((x,y), dest + app_id + '.obj')
        aial_d[app_id] = y
    save_pickle(aial_d, aial_dest)

def make_dataset_generator(aial, batch_size, samples_fd, use_random = False):
    xs_now = []
    ys_now = []

    while True:
        aial_list = [(app_id, cate) for app_id,cate in aial.items()]
        if use_random:
            random.shuffle(aial_list)
        for app_id, _ in aial_list:
            img, cate = load_pickle(samples_fd + app_id + '.obj')
            xs_now.append(img)
            ys_now.append(cate)

            if len(xs_now) == batch_size:
                yield np.array(xs_now), np.array(ys_now)
                xs_now = []
                ys_now = []


if __name__ == '__main__':
    # extract_hog16_sc_from_file('basic_features_t/hog16_sc_test_k3.gzip',
    #     'basic_features_t/hog16_sc_test_k3/',
    #     'basic_features_t/aial_hog16_sc_test_k3.obj')
    
    # aial_train = load_pickle('basic_features_t/aial_hog16_sc_train_k3.obj')
    # aial_test  = load_pickle('basic_features_t/aial_hog16_sc_test_k3.obj')
    # input(aial_test)

    # train_gen = make_dataset_generator(aial_train, 24, 'basic_features_t/hog16_sc_train_k3/', use_random=True)
    # test_gen = make_dataset_generator(aial_test, 24, 'basic_features_t/hog16_sc_test_k3/', use_random=False)
    # m = make_model(feature = 'hog', input_shape=9072)
    # m.fit_generator(train_gen, steps_per_epoch = math.ceil(len(aial_train)/24),
    #     validation_data = test_gen,
    #     validation_steps = math.ceil(len(aial_test)/24),
    #     epochs=10)

    # aial = load('aial_seed_327.obj')
    # split_train_test('basic_features_t/sc_hog16.gzip', 
    #     train_path='basic_features_t/hog16_sc_train_k2.gzip',
    #     test_path ='basic_features_t/hog16_sc_test_k2.gzip', k_iter = 2, aial_obj=aial, sc=True)

    aial = load('aial_seed_327.obj')
    # for i in range(4):
    #     split_train_test('basic_features_t/gist_icon_t.obj', 
    #         train_path='journal/basic_features/gist_icon_train_k%d.gzip' % (i,),
    #         test_path ='journal/basic_features/gist_icon_test_k%d.gzip' % (i,), k_iter = i, aial_obj=aial, sc=False)

    # check_gist_has_all_icons('basic_features_t/gist_icon_t.txt', app_ids)
    # make_not_computed_gist_sc_list('basic_features_t/gist.sc.txt', os.listdir('screenshots.256.distincted.rem.human'), app_ids)

    # make_gist_obj('basic_features_t/gist.sc.txt', 'basic_features_t/gist_sc_t.obj')
    # o = load('basic_features_t/gist_sc_t.obj')
    # print(o['jp.danball.valistroke11.png'])

    #hog stuff
    # model = load_model('journal/basic_features/models/hog16_icon_model1_k3_t-ep-476-loss-0.080-acc-0.714-vloss-2.941-vacc-0.316.hdf5')

    # def extract_fn(img):
    #     return extract_hog(img, pixels_per_cell=(16,16))

    # aial_train, aial_test = split_aial(aial, k_iter=3)
    # test_set = make_test_set(aial_test, samples_fd='icons.combine.recrawled/', resize_w=180, resize_h=180, extract_fn=extract_fn)

    # result = model.evaluate(test_set[0], test_set[1])
    # keras_util.eval_top5(model, test_set[0], test_set[1])

    #gist stuff
    model = load_model('journal/basic_features/models/gist_sc_model7_k3_t-ep-980-loss-0.074-acc-0.717-vloss-2.981-vacc-0.284.hdf5')
    test_fn = 'journal/basic_features/gist_sc/gist_sc_test_k3.obj'
    xtrain, xtest, ytrain, ytest = load_dataset(test_fn, test_fn)

    keras_util.eval_top5(model, xtest, ytest)

