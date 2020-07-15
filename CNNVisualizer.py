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
import gc

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

def load_computed_aial():
    output = {}
    fd = 'visualize_cnn/gradcam/'
    for cate_fd in os.listdir(fd):
        for icon_fn in os.listdir(fd + cate_fd + '/train/'):
            app_id = icon_fn.split('_')[-1][:-4]
            output[app_id] = True
    return output


if __name__ == '__main__':
        model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
        # model_path = 'sim_search_t/models/sc_model2.3_k3_no_aug-ep-085-loss-0.786-acc-0.761-vloss-2.568-vacc-0.403.hdf5'
        sc_dataset_fd = 'c:/screenshots.resized/'
        visualize_fd = None#'visualize_cnn/board/'
        explainer_fn = None#make_vanilla_grad_explain_fn()
        visualize_fn = visualize_grad_cam
        # use_only_selected_cate = True#if False means dont predict from cate
        is_sc = False
        count_max = None
        use_grayscale = False
        use_custom_gradcam = False
        show_visualize = False
        compute_color_magnitude = False
        min_conf = 0

        #auto declare
        count = 0
        magnitude_ratios = [] if explainer_fn is None else None

        computed_aials = load_computed_aial()

        aial = load_pickle('aial_seed_327.obj')
        m = load_model(model_path)
        aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=3, n_splits=4)
        app_id_pred_d = []
        random.shuffle(aial_train)
        
        for x in aial_train:
            app_id = x[0]
            if app_id in computed_aials:
                print('computed already')
                continue
            real_cate_index = np.array(x[2]).argmax()

            #show only or skip game from CATE..
            # if use_only_selected_cate and cates[real_cate_index] != filter_cate:
            #     continue
            # if not use_only_selected_cate and cates[real_cate_index] == filter_cate:
            #     continue

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
                
            #show predict stats
            cate_index = np.argmax(pred[0])
            pred_tuple = []
            for pred_val, cate_str in zip(pred[0], cates):
                pred_tuple.append( (cate_str,pred_val))
            pred_tuple = sorted(pred_tuple, key = lambda x: x[1], reverse=True)
            print(pred_tuple[:5])

            #visualize if not reach count max
            if count_max is None or count < count_max:
                count += 1
                print('saved visualize', app_id, 'count', count)
                if explainer_fn is None:
                    if visualize_fd is not None:
                        save_dest = 'visualize_cnn'
                    else:
                        save_dest = None

                    result = visualize_fn(m, normed_img, cate_index,
                        save_dest = 'visualize_cnn/gradcam/%s/train/%.3f_%s_%s.png' 
                            % (cates[cate_index], conf, cates[real_cate_index], app_id),
                        use_custom_gradcam=use_custom_gradcam,
                        show_visualize=show_visualize,
                        compute_color_magnitude=compute_color_magnitude)
                    if compute_color_magnitude:
                        magnitude_ratios.append(result)
                    
                else:
                    visualize_with_explain_fn(m, normed_img, cate_index, explainer_fn)
            else:
                break
        
        #show magnitude_ratio_avg
        if compute_color_magnitude and magnitude_ratios is not None:
            print('magnitude ratio avg', np.array(magnitude_ratios).mean(axis=0))
            print('magnitude ratio std', np.array(magnitude_ratios).std(axis=0))


        
    



