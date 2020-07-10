from global_util import load_pickle, save_pickle
from icon_util import load_icon_by_fn, rgb_to_gray
from tensorflow.keras.models import load_model, Model
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import keras_util
from tf_explain_util import visualize_grad_cam, make_vanilla_grad_explain_fn, visualize_with_explain_fn
import random
import os

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

if __name__ == '__main__':
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    # model_path = 'sim_search_t/models/sc_model2.3_k3_no_aug-ep-085-loss-0.786-acc-0.761-vloss-2.568-vacc-0.403.hdf5'
    sc_dataset_fd = 'c:/screenshots.resized/'
    visualize_fd = None#'visualize_cnn/board/'
    explainer_fn = None#make_vanilla_grad_explain_fn()
    visualize_fn = visualize_grad_cam
    filter_cate = 'WORD'
    use_only_selected_cate = True#if False means dont predict from cate
    is_sc = False
    count_max = None
    use_grayscale = False
    use_custom_gradcam = False
    show_visualize = True
    min_conf = 0.8

    #auto declare
    count = 0
    magnitude_ratios = [] if explainer_fn is None else None

    aial = load_pickle('aial_seed_327.obj')
    m = load_model(model_path)
    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=3, n_splits=4)
    app_id_pred_d = []
    random.shuffle(aial_test)
    for x in aial_test:
        app_id = x[0]
        real_cate_index = np.array(x[2]).argmax()

        #show only or skip game from CATE..
        if use_only_selected_cate and cates[real_cate_index] != filter_cate:
            continue
        if not use_only_selected_cate and cates[real_cate_index] == filter_cate:
            continue

        if not is_sc:
            img_path = 'icons.combine.recrawled/' + app_id + '.png'
            normed_img = load_icon_by_fn(img_path, 128, 128)/255
        else:
            #find all exists sc fns of a app_id
            exists_sc_fns = []
            for i in range(21):
                sc_fn = '%s%2d.png' % (app_id, i)
                if os.path.exists(sc_dataset_fd + sc_fn):
                    exists_sc_fns.append(sc_dataset_fd + sc_fn)
                else:
                    break
            if len(exists_sc_fns)==0:
                continue
            img_path = random.choice(exists_sc_fns)
            normed_img = load_icon_by_fn(img_path, 256, 160, True)/255

        if use_grayscale:
            normed_img = rgb_to_gray(normed_img)

        pred = m.predict(np.array([normed_img]))
        conf = max(pred[0])

        #show only conf > min_conf and predict selected CATE
        if conf >= min_conf and cates[np.argmax(pred[0])] == filter_cate: pass
        else: continue
            
        app_id_pred_d.append( (app_id,conf))

        #show predict stats
        cate_index = np.argmax(pred[0])
        print('pred cate', cate_index)
        pred_tuple = []
        for pred_val, cate_str in zip(pred[0], cates):
            pred_tuple.append( (cate_str,pred_val))
        pred_tuple = sorted(pred_tuple, key = lambda x: x[1], reverse=True)
        print(cates[real_cate_index])
        print(pred_tuple[:5])

        #visualize if not reach count max
        if count_max is None or count < count_max:
            count += 1
            print('saved visualize', app_id, 'count', count)
            if explainer_fn is None:
                if visualize_fd is not None:
                    save_dest = visualize_fd + app_id + '.png'
                else:
                    save_dest = None

                magnitude_ratio = visualize_fn(m, normed_img, cate_index,
                    save_dest = save_dest,
                    use_custom_gradcam=use_custom_gradcam,
                    show_visualize=show_visualize)
                magnitude_ratios.append(magnitude_ratio)
                
            else:
                visualize_with_explain_fn(m, normed_img, cate_index, explainer_fn)
        else:
            break
    
    #show magnitude_ratio_avg
    if magnitude_ratios is not None:
        print('magnitude ratio avg', np.array(magnitude_ratios).mean(axis=0))
        print('magnitude ratio std', np.array(magnitude_ratios).std(axis=0))


    # sorted_preds = sorted(app_id_pred_d, key = lambda x : x[1], reverse=True)
    # print(sorted_preds[:10])
    # save_pickle(sorted_preds, 'sorted_preds_icon_model2.4_k3.obj')

    # obj = load_pickle('sorted_preds_icon_model2.4_k3.obj')
    # count = 0
    # for app_id, conf in obj:
    #     if conf <= 0.95 and conf >= 0.9:
    #         print(app_id)


    


        
    



