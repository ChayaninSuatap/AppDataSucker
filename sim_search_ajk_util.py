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

        #create sc folder
        if not os.path.exists(sc_dataset_fd + type_fd):
            os.mkdir(sc_dataset_fd + type_fd)

        for icon_fn in os.listdir(icon_dataset_fd + type_fd):

            # get app_id
            app_id = icon_fn[:-4]
            try:
                app_id_index = int(icon_fn[:-4])
                o = load_pickle(human_app_ids_path)
                app_id = o[app_id_index][0]
            except:
                app_id = icon_fn[:-4]
                app_id_index = app_id

            # copy all sc

            for i in range(21):
                sc_name = '%s%2d.png' % (app_id, i)
                try:
                    sc_path = sc_dataset_fd + type_fd + '/' + ('%d%2d.png') % (app_id_index, i)
                except:
                    sc_path = sc_dataset_fd + type_fd + '/' + ('%s%2d.png') % (app_id_index, i)

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
    found_at_count = np.zeros(( len(img_d)-1,))
    for k,v in mrr_d.items():
        main_type = type_d[k]
        for i,(img_b, dis) in enumerate(v):
            found_at = i + 1
            if main_type == type_d[img_b]:
                total_mrr += 1 / found_at
                added_count += 1
                found_at_count[found_at-1]+=1
                break

    mrr = total_mrr / len(mrr_d)
    print(mrr)

    return type_d, mrr_d, found_at_count

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
                # print('pred',fn)
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
        
        #keep game distinct
        for k,v in mrr_d.items():
            a = mrr_d[k]
            output = []
            while len(a)>0:
                cur = a.pop(0)
                output.append(cur)
                for i in range(len(a)-1, -1, -1):
                    if a[i][0][:-2] == cur[0][:-2]:
                        a.remove(a[i])
            mrr_d[k] = output

    if save_cache:
        save_pickle((type_d, img_d, mrr_d, dis_d),pred_cache_path)
        print('save done')
    
    #compute mrr
    total_mrr = 0
    added_count = 0
    found_at_count = np.zeros(( 24,))
    for k,v in mrr_d.items():
        main_type = type_d[k]
        for i,(img_b, dis) in enumerate(v):
            found_at = i + 1
            if main_type == type_d[img_b]:
                total_mrr += 1 / found_at
                added_count += 1
                found_at_count[found_at-1] += 1

                if found_at == 1:
                    if k[:-2] == img_b[:-2]:
                        raise ValueError('suggest it own game!')
                break

    mrr = total_mrr / len(mrr_d)
    print(mrr)
    return mrr_d, found_at_count

def compute_mrr(mrr_d, type_d):
    #compute mrr
    total_mrr = 0
    added_count = 0
    found_at_count = np.zeros((len(mrr_d),))
    for k,v in mrr_d.items():
        main_type = type_d[k]
        for i,(img_b, dis) in enumerate(v):
            found_at = i + 1
            if main_type == type_d[img_b]:
                total_mrr += 1 / found_at
                added_count += 1
                found_at_count[found_at-1] += 1
                break

    mrr = total_mrr / len(mrr_d)
    print(mrr)
    return type_d, mrr_d, found_at_count

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

def plot_pca(icon_cache_path, print_text = True, col_d = 
    {'blocky':'red', 'card':'blue', 'driving_sim':'orange', 'war':'magenta', 'words':'green'},
    auto_gen_col = False, plot_by_game = False, title='', label_d = None):
    type_d, img_d, mrr_d, _ = load_pickle(icon_cache_path) 

    data = np.array(list(img_d.values()))
    result = clustering_util.make_pca(data)

    pbg_col = {}
    auto_gen_col_d = {}

    for (x,y), icon_fn in zip(result, img_d.keys()):

        if print_text:
            plt.text(x,y, icon_fn, {'size':8})

        label = type_d[icon_fn] if label_d == None else label_d[type_d[icon_fn]]

        if plot_by_game:
            app_id = icon_fn[:-6]
            if app_id not in pbg_col:
                pbg_col[app_id] = np.random.rand(3,)
            plt.scatter(x,y, s = 50, color = pbg_col[app_id], label = app_id)
        elif auto_gen_col:
            if type_d[icon_fn] not in auto_gen_col_d:
                auto_gen_col_d[type_d[icon_fn]] = np.random.rand(3,)
            plt.scatter(x,y, s = 50, color = auto_gen_col_d[type_d[icon_fn]], label = label) 
        else:
            plt.scatter(x,y, s = 50, color = col_d[type_d[icon_fn]], label = label)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.show()

def plot_found_at_count(line, title, limit_x = 9999):
    from pylab import rcParams
    rcParams['figure.figsize'] = 9, 5
    splited = list(map(float,line.split('\t')))[:limit_x]
    x = list(map(str, range(1,limit_x+1)))
    plt.bar(x,splited)
    plt.xlabel('Nearest correct suggestion', fontsize = 12) 
    plt.ylabel('Frequency', fontsize = 12)
    plt.title(title, fontsize=12)
    plt.show()

if __name__ == '__main__':
    icon_dataset_fd = 'sim_search_ajk/dataset/icon_puzzle/'
    sc_dataset_fd = 'sim_search_ajk/dataset/sc_puzzle/'
    sc_fd = 'C:/screenshots.resized/'
    # copy_sc_dataset_base_on_icon(icon_dataset_fd, sc_dataset_fd, sc_fd)
    # input()

    icon_model_paths = [
        'sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5', #0.39922222222222226 #pca 0.620
        # 'sim_search_t/models/icon_model2.4_k1_t-ep-497-loss-0.273-acc-0.912-vloss-3.597-vacc-0.370.hdf5', #0.45232034632034623 #pca 0.640
        # 'sim_search_t/models/icon_model2.4_k2_t-ep-463-loss-0.283-acc-0.904-vloss-3.585-vacc-0.368.hdf5', #0.5571528822055138 #pca 0.571
        # 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5' #0.47025526107879045 #pca 0.635
        #0.6169
    ]

    sc_model_paths = [
        'sim_search_t/models/sc_model2.3_k0_no_aug-ep-087-loss-0.721-acc-0.782-vloss-2.793-vacc-0.400.hdf5',  #0.4112483636213254 #pca 0.6438055771389106
        # 'sim_search_t/models/sc_model2.3_k1_no_aug-ep-135-loss-0.995-acc-0.708-vloss-2.763-vacc-0.398.hdf5', #0.5123068412801622 #pca 0.8075961636567699
        # 'sim_search_t/models/sc_model2.3_k2_no_aug-ep-068-loss-0.907-acc-0.723-vloss-2.568-vacc-0.396.hdf5', #0.4684522538516435 #pca 0.557 0.6155817433595208
        # 'sim_search_t/models/sc_model2.3_k3_no_aug-ep-085-loss-0.786-acc-0.761-vloss-2.568-vacc-0.403.hdf5' #0.4203212543079958 #pca 0.622 0.6826307811156297
    ]

    #icon found_at_count average
    #10.25	6.25	3.5	1.75	0.25	0	0.25	0.25	0.25	0.25	0.25	0.5	0.25	0.5	0	0.25	0	0.25	0	0	0	0	0	0
    #sc case
    #154.5	54.5	41.75	15	10.5	6.75	2	4	3.5	0.75	1.25	0.5	0.25	0	0.25	0	0	0.25	0	1.25	0	0	0	0
    #icon + sc case
    #16.25	3.5	2	1.25	0.5	1	0.25	0.25

    # plot_found_at_count('16.25	3.5	2	1.25	0.5	1	0.25	0.25',
        # 'Frequency of nearest correct suggestion using model I10 and model S9', limit_x = 8)

    save_cache = False
    load_cache = True

    icon_cache_path = 'sim_search_ajk/dataset/icon_k0_puzzle.obj'
    type_d, mrr_d, found_at_count = compute_icon_to_icon_distance(icon_dataset_fd, icon_model_paths, distance_fn = euclidean, pred_cache_path = icon_cache_path, save_cache = save_cache, load_cache = load_cache, use_pca = True)
    print(found_at_count)

    sc_cache_path = 'sim_search_ajk/dataset/sc_k0_puzzle.obj'
    mrr_d, found_at_count = compute_sc_to_sc_distance(sc_dataset_fd, sc_model_paths, distance_fn = euclidean, pred_cache_path = sc_cache_path, load_cache = load_cache, save_cache = save_cache, use_pca = True)
    print(found_at_count)
    
    # show_kmeans_clustering(sc_cache_path)
    five_labels_d = {'blocky':'Blocky', 'card':'Card', 'driving_sim':'Driving simulator', 'war':'War','words':'Word'}
    sports_labels_d = {'basketball':'Basketball', 'football':'Football', 'snooker':'Snooker'}
    sports_col_d = {'basketball':'magenta', 'football':'blue', 'snooker':'red'}
    puzzle_col_d = {'Blocky':'red', 'Number':'orange', 'Word':'blue'}
    # plot_pca(icon_cache_path, print_text = False,
        # auto_gen_col=False, plot_by_game=False, title='Model I10 feature visualization', label_d=five_labels_d)

    type_d, mrr_d, found_at_count = compute_game_to_game_distance(icon_cache_path, sc_cache_path)
    print(found_at_count)
    # write_rank_into_file(type_d = type_d, mrr_d = mrr_d)
