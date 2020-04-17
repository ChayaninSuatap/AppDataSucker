import os
from global_util import load_pickle, save_pickle
from shutil import copyfile
from similarity_search_util import mod_model, euclidean
from icon_util import load_icon_by_fn
import numpy as np

def copy_sc_dataset_base_on_icon(icon_dataset_fd, sc_dataset_fd, sc_fd, human_app_ids_path = 'app_ids_for_human_test.obj'):
    for type_fd in os.listdir(icon_dataset_fd):
        for icon_fn in os.listdir(icon_dataset_fd + type_fd):

            # get app_id
            app_id = icon_fn[:-4]
            app_id_index = int(icon_fn[:-4])
            o = load_pickle(human_app_ids_path)
            app_id = o[app_id_index][0]

            # copy all sc

            for i in range(21):
                sc_name = '%s%2d.png' % (app_id, i)
                sc_path = sc_dataset_fd + type_fd + '/' + ('%d%2d.png') % (app_id_index, i)
                try:
                    copyfile(sc_fd + sc_name, sc_path)
                    print(sc_name, sc_path)
                except:
                    pass

def compute_icon_to_icon_distance(icon_dataset_fd, model_path, distance_fn):
    model = mod_model(model_path)
    type_d = {} # fn --> type that I splited myself
    dis_d = {} # fn1_fn2 distance
    img_d = {} # fn --> feature vector
    mrr_d = {} # fn --> [(comparing_fn,distance), (,), (,), ...]

    #make type_d & predict & save feature vector & init mrr_d keys
    for type_fd in os.listdir(icon_dataset_fd):
        for icon_fn in os.listdir(icon_dataset_fd + type_fd + '/'):
            fn = icon_fn[:-4]
            type_d[fn] = type_fd

            icon_path = icon_dataset_fd + type_fd + '/' + icon_fn
            icon = load_icon_by_fn(icon_path, 128, 128)
            icon = np.array([icon]) / 255
            img_d[fn] = model.predict(icon)[0]

            mrr_d[fn] = []
    
    #make dis_d and mrr_d
    for img_a in img_d.keys():
        for img_b in img_d.keys():
            if img_a == img_b: continue

            dis = distance_fn(img_d[img_a], img_d[img_b])
            dis_d[ img_a + '_' + img_b ] = dis

            mrr_d[img_a].append( (img_b,dis) )
    
    #sort mrr_d
    for k,v in mrr_d.items():
        mrr_d[k] = sorted(v, key = lambda x: x[1])

    #compute mrr
    total_mrr = 0
    added_count = 0
    for k,v in mrr_d.items():
        main_type = type_d[k]
        for i,(img_b, dis) in enumerate(v):
            found_at = i + 1
            if main_type == type_d[img_b]:
                total_mrr += 1 / found_at
                added_count += 1
                break

    mrr = total_mrr / len(mrr_d)
    print(mrr)
    print(added_count)
    print(len(mrr_d))



if __name__ == '__main__':
    icon_dataset_fd = 'sim_search_ajk/dataset/icon/'
    sc_dataset_fd = 'sim_search_ajk/dataset/sc/'
    sc_fd = 'screenshots/'
    # copy_sc_dataset_base_on_icon(icon_dataset_fd, sc_dataset_fd, sc_fd)

    # model_path = 'sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5' #39
    # model_path = 'sim_search_t/models/icon_model2.4_k1_t-ep-497-loss-0.273-acc-0.912-vloss-3.597-vacc-0.370.hdf5'   #45
    model_path = 'sim_search_t/models/icon_model2.4_k2_t-ep-463-loss-0.283-acc-0.904-vloss-3.585-vacc-0.368.hdf5'   #55
    # model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5' #47
    compute_icon_to_icon_distance(icon_dataset_fd, model_path,
        distance_fn = euclidean)
