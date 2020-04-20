import os
from global_util import load_pickle, save_pickle
from shutil import copyfile
from similarity_search_util import mod_model, euclidean, mse, sub_and_sum
from icon_util import load_icon_by_fn
import numpy as np
import clustering_util
import matplotlib.pyplot as plt
from collections import OrderedDict

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

def compute_icon_to_icon_distance(icon_dataset_fd, model_paths, distance_fn, pred_cache_path,
    load_cache=False, save_cache=False, use_pca = False):
    #load models
    models = []
    for model_path in model_paths:
        models.append(mod_model(model_path))
        
    type_d = {} # fn --> type that I splited myself
    dis_d = {} # fn1_fn2 distance
    img_d = {} # fn --> feature vector
    mrr_d = {} # fn --> [(comparing_fn,distance), (,), (,), ...]

    if load_cache:
        type_d, img_d, mrr_d, dis_d = load_pickle(pred_cache_path)
    else:
        #make type_d & predict & save feature vector & init mrr_d keys
        for type_fd in os.listdir(icon_dataset_fd):
            for icon_fn in os.listdir(icon_dataset_fd + type_fd + '/'):
                fn = icon_fn[:-4]
                type_d[fn] = type_fd

                icon_path = icon_dataset_fd + type_fd + '/' + icon_fn
                icon = load_icon_by_fn(icon_path, 128, 128)
                icon = np.array([icon]) / 255

                pred = models[0].predict(icon)[0]
                for model in models[1:]:
                    pred += model.predict(icon)[0]
                img_d[fn] = pred / len(models)

                mrr_d[fn] = []

        #transform into 2 dimension
        if use_pca:
            data = np.array(list(img_d.values()))
            result = clustering_util.make_pca(data)
            for img_d_key, e in zip(img_d.keys(), result):
                img_d[img_d_key] = e
        
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
    
    if save_cache:
        save_pickle((type_d, img_d, mrr_d, dis_d),pred_cache_path)
        print('save done') 

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

    return type_d, mrr_d

def compute_sc_to_sc_distance(sc_dataset_fd, model_paths, distance_fn, pred_cache_path,
    load_cache = False, save_cache = False, use_pca = False):
    #load models
    models = []
    for model_path in model_paths:
        models.append(mod_model(model_path))

    type_d = {} # fn --> type that I splited myself
    dis_d = {} # fn1_fn2 distance
    img_d = {} # fn --> feature vector
    mrr_d = {} # fn --> [(comparing_fn,distance), (,), (,), ...]

    #make type_d & predict & save feature vector & init mrr_d keys
    if load_cache:
        type_d, img_d, mrr_d, dis_d = load_pickle(pred_cache_path)
    else:
        for type_fd in os.listdir(sc_dataset_fd):
            for sc_fn in os.listdir(sc_dataset_fd + type_fd + '/'):
                fn = sc_fn[:-4]
                type_d[fn] = type_fd

                sc_path = sc_dataset_fd + type_fd + '/' + sc_fn
                sc = load_icon_by_fn(sc_path, 256, 160, rotate_for_sc = True)
                sc= np.array([sc]) / 255

                pred = models[0].predict(sc)[0]
                print('pred',fn)
                for model in models[1:]:
                    pred += model.predict(sc)[0]
                img_d[fn] = pred / len(models)

                mrr_d[fn] = []
        
        #transform into 2 dimension
        if use_pca:
            data = np.array(list(img_d.values()))
            result = clustering_util.make_pca(data)
            for img_d_key, e in zip(img_d.keys(), result):
                img_d[img_d_key] = e

        #make dis_d and mrr_d
        for img_a in img_d.keys():
            for img_b in img_d.keys():
                if img_a == img_b: continue
                if img_a[:-2] == img_b[:-2] :continue

                dis = distance_fn(img_d[img_a], img_d[img_b])
                dis_d[ img_a + '_' + img_b ] = dis

                mrr_d[img_a].append( (img_b,dis) )
    
            #sort mrr_d
        for k,v in mrr_d.items():
            mrr_d[k] = sorted(v, key = lambda x: x[1])

    if save_cache:
        save_pickle((type_d, img_d, mrr_d, dis_d),pred_cache_path)
        print('save done')
    
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
    return mrr_d

def compute_mrr(mrr_d, type_d):
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
    return type_d, mrr_d

def compute_game_to_game_distance(icon_cache_path, sc_cache_path):

    type_icon, _, _, dis_icon = load_pickle(icon_cache_path)
    type_sc, _, _, dis_sc = load_pickle(sc_cache_path)

    mrr_game = {}
    for ga in type_icon.keys():

        mrr_game[ga] = []

        for gb in type_icon.keys():
            if ga == gb: continue
            #add icon dis
            total_dis = dis_icon[ga + '_' + gb]

            #add sc dis
            for ga_sc_fn in type_sc.keys():
                if ga_sc_fn[:-2] != ga: continue

                min_dis = 9999999

                for gb_sc_fn in type_sc.keys():
                    if gb_sc_fn[:-2] != gb: continue

                    min_dis = min(dis_sc[ga_sc_fn + '_' + gb_sc_fn], min_dis)

                total_dis += min_dis

            mrr_game[ga].append((gb, total_dis))
    
    #sort mrr
    for k,v in mrr_game.items():
        mrr_game[k] = sorted(v, key = lambda x: x[1])

    return compute_mrr(mrr_game, type_icon)

def write_rank_into_file(cache_path = None, output_path = 't.txt', type_d = None, mrr_d = None):
    if cache_path != None:
        type_d, _, mrr_d, _ = load_pickle(cache_path)  
    
    f = open(output_path, 'w')
    for k,v in mrr_d.items():
        print(k, end = ' ', file = f)
        print(type_d[k], end = ' ', file = f)
        for x in v:
            print(x[0], type_d[x[0]], end = ' ', file = f)
        print('', file = f)
    f.close()

def show_kmeans_clustering(icon_cache_path):
    type_d, img_d, _, _ = load_pickle(icon_cache_path) 

    data = np.array(list(img_d.values()))
    idx = clustering_util.kmeans_clustering(data, 5)

    groups = {}
    for group_i, icon_fn in zip(idx, img_d.keys()):
        if group_i not in groups:
            groups[group_i] = []
        
        groups[group_i].append((icon_fn, type_d[icon_fn]))
    
    print(groups)
    for group in groups.values():
        for e in group:
            print(e[0], e[1], end = ' ')
        print()

def plot_pca(icon_cache_path, print_text = True):
    type_d, img_d, mrr_d, _ = load_pickle(icon_cache_path) 

    data = np.array(list(img_d.values()))
    result = clustering_util.make_pca(data)
    col_d = {'blocky':'red', 'card':'blue', 'driving_sim':'yellow',
    'war':'green', 'words':'black'}
    for (x,y), icon_fn in zip(result, img_d.keys()):
        if print_text:
            plt.text(x,y, type_d[icon_fn], {'size':15})
        plt.scatter(x,y, s = 50, color = col_d[type_d[icon_fn]], label = type_d[icon_fn])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

if __name__ == '__main__':
    icon_dataset_fd = 'sim_search_ajk/dataset/icon/'
    sc_dataset_fd = 'sim_search_ajk/dataset/sc/'
    sc_fd = 'screenshots/'
    # copy_sc_dataset_base_on_icon(icon_dataset_fd, sc_dataset_fd, sc_fd)

    model_paths = [
        # 'sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5', #0.39922222222222226 #pca 0.620
        # 'sim_search_t/models/icon_model2.4_k1_t-ep-497-loss-0.273-acc-0.912-vloss-3.597-vacc-0.370.hdf5', #0.45232034632034623 #pca 0.640
        # 'sim_search_t/models/icon_model2.4_k2_t-ep-463-loss-0.283-acc-0.904-vloss-3.585-vacc-0.368.hdf5', #0.5571528822055138 #pca 0.571
        # 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5' #0.47025526107879045 #pca 0.635

        # 'sim_search_t/models/sc_model2.3_k0_no_aug-ep-087-loss-0.721-acc-0.782-vloss-2.793-vacc-0.400.hdf5',  #0.4112483636213254 #pca 0.582
        'sim_search_t/models/sc_model2.3_k1_no_aug-ep-135-loss-0.995-acc-0.708-vloss-2.763-vacc-0.398.hdf5', #0.5123068412801622 #pca 0.783
        # 'sim_search_t/models/sc_model2.3_k2_no_aug-ep-068-loss-0.907-acc-0.723-vloss-2.568-vacc-0.396.hdf5', #0.4684522538516435 #pca 0.557
        # 'sim_search_t/models/sc_model2.3_k3_no_aug-ep-085-loss-0.786-acc-0.761-vloss-2.568-vacc-0.403.hdf5' #0.4203212543079958 #pca 0.622
    ]

    # icon_cache_path = 'sim_search_ajk/dataset/icon_pc_k2_pca.obj'
    # type_d, mrr_d = compute_icon_to_icon_distance(icon_dataset_fd, model_paths,
        # distance_fn = euclidean, pred_cache_path = icon_cache_path, save_cache = True , load_cache = False, use_pca = True)
    
    # write_rank_into_file(icon_cache_path)
    # show_kmeans_clustering(icon_cache_path)

    sc_cache_path = 'sim_search_ajk/dataset/sc_pc_k1_pca.obj'
    # mrr_d = compute_sc_to_sc_distance(sc_dataset_fd, model_paths,
        # distance_fn = euclidean, pred_cache_path = sc_cache_path, load_cache = False, save_cache = True, use_pca = True)
    
    # show_kmeans_clustering(sc_cache_path)
    plot_pca(sc_cache_path, print_text = False)

    # type_d, mrr_d = compute_game_to_game_distance(icon_cache_path, sc_cache_path)
    # write_rank_into_file(type_d = type_d, mrr_d = mrr_d)
