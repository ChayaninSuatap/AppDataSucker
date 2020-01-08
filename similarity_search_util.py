from tensorflow.keras.models import load_model, Model
from icon_util import load_icon_by_fn
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import db_util
import global_util
from overall_feature_util import _all_game_category

def compute_preds(icon_names, model_name = 'models_top_each_task/i3_k0.hdf5',
    icons_fd_path = 'icons.test_similarity_search/', show_output=False):
    model = load_model(model_name)
    input_layer = None
    output_layer = None
    for layer in model.layers:
        if layer.name == 'input_1':
            input_layer = layer
        #test if weights change
        # elif layer.name == 'conv2d':
        #     print(layer.get_weights()[0][0][0][0])
        elif layer.name == 'my_model_flatten':
            output_layer = layer
            break

    model = Model(input_layer.input, output_layer.output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    
    preds = {}
    for icon_name in icon_names:
        try:
            icon = load_icon_by_fn(icons_fd_path + icon_name, 128, 128)
        except:
            continue
        
        if show_output: print(icon_name)

        pred = model.predict(np.array([icon]) / 255)[0]
        preds[icon_name] = pred
    
    return preds

#mse
def mse(pred1, pred2):
    return ((pred1 - pred2)**2).mean(axis=None)

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
    for i in range(340):
        gt = input()
        gts.append(int(gt))
    global_util.save_pickle(gts, fn)

def compute_screenshot_preds():
    pass

if __name__ == '__main__':

    create_human_testset_groundtruth()
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
