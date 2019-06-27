import sc_util
import random
import numpy as np
import preprocess_util
from keras_util import gen_k_fold_pass
import icon_cate_util
import icon_util
import global_util
import mypath

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
aial_train, aial_test = gen_k_fold_pass(aial, kf_pass=0, n_splits=4)

aial_train_sc, aial_test_sc = sc_util.make_aial_sc(aial_train, aial_test, sc_dict)

#compute app that have icon and screenshot
# valid_app_ids = []
# for app_id, rating, label in aial_test:
#     #try read icon
#     try:
#         icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
#         if app_id in sc_dict:
#             for sc_fn in sc_dict[app_id]:
#                 sc = icon_util.load_icon_by_fn(mypath.screenshot_folder + sc_fn, 256, 160, rotate_for_sc=True)
#                 valid_app_ids.append( app_id)
#                 break
#     except Exception as e:
#         print('failed', repr(e), app_id)
valid_app_ids = global_util.load_pickle('valid_app_ids.obj')
#make dict app_id -> category
app_id_to_cate = {}
for app_id, _, cate in aial_test:
    app_id_to_cate[app_id] = cate
#compute frequency
cate_counter = [0] * 17
#get app_ids_for_human_test
app_ids_for_human_test = []
#dirty hardcode 
from shutil import copyfile
with open('ground_truth.txt', 'w') as ground_truth_f:
    pass
ground_truth_f = open('ground_truth.txt', 'a')
##
for app_id in valid_app_ids:
    #this app_id is gone before downloading 512 size image
    if app_id == 'com.spacegame.wordconnect2': continue
    if app_id == 'com.square_enix.android_googleplay.FFV_GP' : continue
    if app_id == 'com.square_enix.FFIXww.android_googleplay' : continue
    index = app_id_to_cate[app_id].index(1)
    #skip if got 20 already
    if cate_counter[index] < 20:
        app_ids_for_human_test.append((app_id, index))
        #dirty hardcode
        # copyfile('icons/' + app_id +'.png', 'icons_human_test/%d.png' % (fn_i,))
        # ground_truth_f.writelines(str(index)+'\n')
        # fn_i += 1
        cate_counter[index]+=1

random.shuffle(app_ids_for_human_test)
global_util.save_pickle(app_ids_for_human_test, 'app_ids_for_human_test.obj')
for i,(app_id, label) in enumerate(app_ids_for_human_test):
    ground_truth_f.write(str(label)+'\n')
    copyfile('icons/' + app_id +'.png', 'icons_human_test/%d.png' % (i,)) 
print(app_ids_for_human_test)
ground_truth_f.close()





