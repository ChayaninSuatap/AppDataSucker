import os
import mypath
def make_sc_dict():
    sc_dict = {} #map app_id and array of it each sceenshot .png
    for fn in os.listdir(mypath.screenshot_folder):
        app_id = fn[:-6]
        if app_id not in sc_dict:
            sc_dict[app_id] = [fn]
        else:
            sc_dict[app_id].append(fn)
    return sc_dict

def make_aial_sc(aial_train, aial_test, sc_dict):
    aial_train_sc = []
    aial_test_sc = []
    for app_id, rating, cate  in aial_train:
        if app_id not in sc_dict: continue
        for sc_fn in sc_dict[app_id]:
            aial_train_sc.append((sc_fn, rating, cate))
    for app_id, rating, cate  in aial_test:
        if app_id not in sc_dict: continue
        for sc_fn in sc_dict[app_id]:
            aial_test_sc.append((sc_fn, rating, cate))
    print('aial len train and test',len(aial_train_sc), len(aial_test_sc))
    return aial_train_sc, aial_test_sc