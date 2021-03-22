from tensorflow.keras.models import load_model, Model
from icon_util import load_icon_by_fn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import db_util
import global_util
from overall_feature_util import _all_game_category
from global_util import load_pickle, save_pickle
import sc_util
import icon_util

def mod_model(model_path):
    model = load_model(model_path)
    #drop softmax layer
    input_layer = None
    output_layer = None
    for layer in model.layers:
        if layer.name == 'input_1':
            input_layer = layer
        elif layer.name == 'my_model_flatten':
            output_layer = layer
            break
    model = Model(input_layer.input, output_layer.output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def compute_preds(icon_names, model_path,
    icons_fd_path, use_feature_vector, show_output=False):

    model = load_model(model_path)
    #drop softmax layer
    if use_feature_vector:
        input_layer = None
        output_layer = None
        for layer in model.layers:
            if layer.name == 'input_1':
                input_layer = layer
            elif layer.name == 'my_model_flatten':
                output_layer = layer
                break
        model = Model(input_layer.input, output_layer.output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    preds = {}
    icons = []
    icon_names_pred = []

    for icon_name in icon_names:
        try:
            icon = load_icon_by_fn(icons_fd_path + icon_name, 128, 128)
        except:
            continue
        
        if show_output: print(icon_name)

        icons.append(icon)
        icon_names_pred.append(icon_name)
        if len(icons) == 128:
            pred = model.predict(np.array(icons) / 255)
            for p, icon_name in zip(pred, icon_names_pred):
                preds[icon_name] = p
            icons = []
            icon_names_pred = []

    if len(icons) > 0:
        pred = model.predict(np.array(icons) / 255)
        for p, icon_name in zip(pred, icon_names_pred):
            preds[icon_name] = p

    return preds

#mse
def mse(pred1, pred2):
    return ((pred1 - pred2)**2).mean(axis=None)

def sub_and_sum(pred1, pred2):
    return (np.absolute(pred1 - pred2)).sum()

def euclidean(pred1, pred2):
    return np.linalg.norm(pred1 - pred2)

# compute distance of icons pairwise
def compute_distances(icon_names, preds, distance_fn = mse):
    distances = []
    for i in range(0, len(preds)-1):
        for j in range(i+1, len(preds)):
            dis = distance_fn(preds[i], preds[j])
            distances.append( (icon_names[i], icon_names[j], dis))

    #sort by nearest
    distances = sorted(distances, key=  lambda x : x[2])
    return distances

def make_category_dict():
    'app_id(str) -> cate(str)'
    dict = {}
    conn = db_util.connect_db()
    queries = conn.execute('select app_id, category from app_data')
    for row in queries:
        if row[1] is not None:
            splited = row[1].split(',')
            if len(splited) == 2:
                cate = splited[0]
            else:
                cate = splited[0] if 'FAMILY' in splited[1] else splited[1]
            dict[row[0]] = cate
    return dict

def get_human_testset_cate_from_fn(fn):
    human_testset_index = int(fn[:-4])
    human_testset_groundtruth_list = global_util.load_pickle('similarity_search/human_testset_groundtruth_list.obj')
    cate_index = human_testset_groundtruth_list[human_testset_index]
    return _all_game_category[cate_index]

def copy_icons_randomly_for_test():
    icon_fns = list(os.listdir('icons.backup'))
    import random
    import shutil
    random.shuffle(icon_fns)

    cate_dict = make_category_dict()
    icon_fn_cate_dict = {}

    for icon_fn in icon_fns:
        if icon_fn[:-4] in cate_dict and 'GAME' in cate_dict[icon_fn[:-4]]:
            icon_fn_cate_dict[icon_fn] = cate_dict[icon_fn[:-4]]

    #copy each cate by 20 games
    for cate in set(list(icon_fn_cate_dict.values())):
        count = 0
        for k, v in icon_fn_cate_dict.items():
            if v == cate:
                count += 1
                shutil.copyfile('icons.backup/'+k, 'icons.test_similarity_search/'+k)

            if count == 20:
                break

def copy_scs_randomly_for_test():
    input_path = 'screenshots.256.distincted'
    def make_sc_dict():
        sc_dict = {} #map app_id and array of it each sceenshot .png
        for fn in os.listdir(input_path):
            app_id = fn[:-6]
            if app_id not in sc_dict:
                sc_dict[app_id] = [fn]
            else:
                sc_dict[app_id].append(fn)
        return sc_dict
    
    sc_dict = make_sc_dict()
    print(sc_dict)

def create_human_testset_groundtruth():
    fn = 'similarity_search/human_testset_groundtruth_list.obj'
    print('enter human testset groundtruth from spreadsheet :')
    gts = []
    for _ in range(340):
        gt = input()
        gts.append(int(gt))
    global_util.save_pickle(gts, fn)

def compute_mean_preds_caches(preds_caches_fd, preds_caches_fn):
    caches = []
    for fn in preds_caches_fn:
        o = load_pickle(preds_caches_fd + fn)
        caches.append(o)

    output = {}
    for app_id in caches[0].keys():

        sum = caches[0][app_id]

        for cache in caches[1:]:
            sum += cache[app_id]
        
        sum /= len(caches)
        output[app_id] = sum
    
    return output

def create_mean_preds_caches(preds_caches_fd, preds_caches_fn, output_path):
    o = compute_mean_preds_caches(preds_caches_fd, preds_caches_fn)
    save_pickle(o, output_path)

def create_all_sc_human_set_obj(sc_fd ,human_app_ids_path = 'app_ids_for_human_test.obj'):
    sc_dict = sc_util.make_sc_dict(sc_fd)
    o = load_pickle(human_app_ids_path)
    
    count_app_id_error = 0
    output = {}
    for app_id, _ in o:

        if app_id in sc_dict:
            
            for sc_fn in sc_dict[app_id]:
                try:
                    sc = icon_util.load_icon_by_fn(sc_fd + sc_fn, 256, 160, rotate_for_sc = True)
                    sc = sc.astype('float32') / 255
                    output[sc_fn] = sc
                    print(sc_fn)
                except Exception as e:
                    print(repr(e))
        else:
            count_app_id_error += 1
    

    save_pickle( output, 'all_sc_human_testset.obj')
    print('len output', len(output))
    print('error', count_app_id_error)

def add_topn(a, b, img_name, dis, topn, is_icon = False, is_sc = False):

    app_id = img_name[:-4] if is_icon else img_name[:-6]

    if len(a) < topn and app_id not in b:
        a.append( (img_name, dis))
        b[app_id] = True
    elif a[-1][1] > dis and app_id not in b:
        a.append( (img_name, dis))
        a.sort(key = lambda x:x[1])
        b[app_id] = True

    while(len(a)>topn):
        forsaked_img_name = a.pop()[0]
        if forsaked_img_name[:-4] in b:
            del b[forsaked_img_name[:-4]]
        else:
            del b[forsaked_img_name[:-6]]

def get_topn_using_icon_and_screenshot(icon_pc_path ,icon_hpc_path, sc_pc_path, sc_hpc_path, distance_fn,
    topn, human_app_ids_path = 'app_ids_for_human_test.obj', bypass_icon=False):

    icon_pc = load_pickle(icon_pc_path)
    icon_hpc = load_pickle(icon_hpc_path)

    sc_pc = load_pickle(sc_pc_path)
    sc_hpc = load_pickle(sc_hpc_path)

    output = {}

    for i,(app_id,_) in enumerate(load_pickle(human_app_ids_path)):
        
        global_topn = []
        global_topn_d = {}

        icon_key = str(i) + '.png'
        print('making', icon_key)
        #find top n icon
        if not bypass_icon:
            main_icon = icon_hpc[icon_key]
            for icon_name, icon_v in icon_pc.items():
                dis = distance_fn(main_icon, icon_v)
                add_topn(global_topn, global_topn_d, icon_name, dis, topn=topn, is_icon=True)
        
        #find top n sc
        for sc_human_name in sc_hpc.keys():
            #filter sc_fn from human set that belong to app_id
            if sc_human_name[:-6] == app_id:
                main_sc = sc_hpc[sc_human_name]

                print('making', sc_human_name)

                for sc_name, sc_v in sc_pc.items():
                    dis = distance_fn(main_sc, sc_v)
                    add_topn(global_topn, global_topn_d, sc_name, dis, topn=topn, is_sc = True)

        output[icon_key] = global_topn
        print(icon_key, output[icon_key])
    return output

def get_topn_using_screenshot(sc_pc_path, sc_hpc_path, distance_fn, topn):

    sc_pc = load_pickle(sc_pc_path)
    sc_hpc = load_pickle(sc_hpc_path)

    output = {}

    for i,(_, sc_main) in enumerate(sc_hpc.items()):
        
        global_topn = []
        global_topn_d = {}
        app_id = str(i) + '.png'
        
        #find top n sc
        for sc_name, sc_v in sc_pc.items():
            dis = distance_fn(sc_main, sc_v)
            add_topn(global_topn, global_topn_d, sc_name, dis, topn=topn, is_sc = True)

        output[app_id] = global_topn
        print(app_id, output[app_id])
    return output

def get_topn_using_icon(icon_pc_path, icon_hpc_path, distance_fn, topn):
    icon_pc = load_pickle(icon_pc_path)
    icon_hpc = load_pickle(icon_hpc_path)
    output = {}
    for app_id, main_icon in icon_hpc.items():
        global_topn = []
        global_topn_d = {}
        
        for icon_name, icon_v in icon_pc.items():
            dis = distance_fn(main_icon, icon_v)
            add_topn(global_topn, global_topn_d, icon_name, dis, topn=topn, is_icon = True)
        
        output[app_id] = global_topn
        print(app_id, output[app_id])
    return output

def filter_non_original_sc_human_set_from_sc_hpc(sc_hpc_path, sc_hpc_output_path, sc_human_testset_obj_path = 'sc_human_testset.obj'):
    sc_hpc = load_pickle(sc_hpc_path)
    sc_human_testset = load_pickle(sc_human_testset_obj_path)
    output = {}
    for k,v in sc_hpc.items():
        if k in sc_human_testset:
            output[k] = v
            print(k)
    save_pickle(output, sc_hpc_output_path)


if __name__ == '__main__':
    import csv
    for k_iter in [0,1,2,3]:
        icon_model_paths = [
            'sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5', #0.39922222222222226 #pca 0.620
            'sim_search_t/models/icon_model2.4_k1_t-ep-497-loss-0.273-acc-0.912-vloss-3.597-vacc-0.370.hdf5', #0.45232034632034623 #pca 0.640
            'sim_search_t/models/icon_model2.4_k2_t-ep-463-loss-0.283-acc-0.904-vloss-3.585-vacc-0.368.hdf5', #0.5571528822055138 #pca 0.571
            'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5' #0.47025526107879045 #pca 0.635
        ]
        
        cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

        aial = load_pickle('aial_seed_327.obj')

        model = mod_model(icon_model_paths[k_iter])

        # create data for ajk
        # # result = {}
        # # for app_id, _, cate, *_ in aial:
        # #     icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
        # #     icon = np.array([icon])
        # #     pred = model.predict(icon)[0]
        # #     cate = cates[cate.index(1)]
        # #     result[app_id] = (cate, pred)
        # # save_pickle(result, 'journal/flatten_preds/icon/k%s.obj' % (k_iter,))

        # # create csv
        # # import csv
        # # with open('journal/flatten_preds/icon/k%s.csv' % (k_iter,), 'w', newline='') as csv_file:
        # #     header = ['app_id', 'category', 'feature']
        # #     writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # #     obj = load_pickle('journal/flatten_preds/icon/k%s.obj' % (k_iter,))
        # #     for app_id, (cate, pred) in obj.items():
        # #         writer.writerow([app_id, cate] + pred.tolist())


    create_all_sc_human_set_obj('screenshots/')
    # create_human_testset_groundtruth()
    # cate_dict = make_category_dict()

    # preds = compute_preds(icon_names)
    # distances = compute_distances(icon_names, preds, distance_fn = mse)

    # fig, ax = plt.subplots(10, 6, figsize=(8,4))
    
    # [x.set_axis_off() for x in ax.ravel()]
    
    # for i in range(10):
    #     img_a_name = distances[i][0]
    #     img_b_name = distances[i][1]
    #     ax[i, 0].imshow(Image.open('icons.test_similarity_search/'+img_a_name))
    #     ax[i, 1].imshow(Image.open('icons.test_similarity_search/'+img_b_name))
    #     ax[i, 2].text(0, 0.5, 'distance=%f \n%s %s' % (distances[i][2], cate_dict[img_a_name[:-4]], cate_dict[img_b_name[:-4]]))
    
    # for i in range(10):
    #     img_a_name = distances[::-1][i][0]
    #     img_b_name = distances[::-1][i][1]
    #     ax[i, 3].imshow(Image.open('icons.test_similarity_search/'+img_a_name))
    #     ax[i, 4].imshow(Image.open('icons.test_similarity_search/'+img_b_name))
    #     ax[i, 5].text(0, 0.5, 'distance=%f \n%s %s' % (distances[::-1][i][2], cate_dict[img_a_name[:-4]], cate_dict[img_b_name[:-4]]))

    # f = open('similarity_search.txt', 'w')
    # for x in distances:
    #     print(x[2], x[0], x[1],  cate_dict[x[0][:-4]], cate_dict[x[1][:-4]], file=f)
    # f.close()
    # plt.show()
