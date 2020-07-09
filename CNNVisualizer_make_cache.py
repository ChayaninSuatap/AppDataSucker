from global_util import load_pickle, save_pickle
from keras_util import gen_k_fold_pass
import os
from tensorflow.keras.models import load_model, Model
from icon_util import load_icon_by_fn
import numpy as np
from tf_explain_util import visualize_grad_cam

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

class Record:

    def __init__(self, app_id, cate_index_real, cate_index_pred,
        pred_conf, magnitude_ratio):
        self.app_id = app_id
        self.cate_index_real = cate_index_real
        self.cate_index_pred = cate_index_pred
        self.pred_conf = pred_conf
        self.magnitude_ratio = magnitude_ratio

if __name__ == '__main__':
    k_fold = 3
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    icon_fd = 'icons.combine.recrawled/'

    aial = load_pickle('aial_seed_327.obj')
    aial_filtered = [(app_id, cate) for app_id, _, cate, *_ in aial]
    _, aial_test = gen_k_fold_pass(aial_filtered, k_fold, 4)
    aial_test_d = {app_id:cate for app_id,cate in aial_test}

    model = load_model(model_path)
    records = []
    for fn in os.listdir(icon_fd):
        app_id = fn[:-4]

        #filter icon not in app_ids
        if app_id not in aial_test_d:
            continue

        img = load_icon_by_fn(icon_fd + fn, 128, 128) /255
        print(app_id)
        pred = model.predict(np.array([img]))

        conf_max = pred[0].max()
        cate_index_pred = pred[0].argmax()
        cate_index_real = np.array(aial_test_d[app_id]).argmax()
        print('real', cates[cate_index_real], 'pred', cates[cate_index_pred], 'conf', conf_max)

        magnitude_ratio = visualize_grad_cam(model, img,
            cate_index_pred, show_visualize=False)
        print('magnitude ratio', magnitude_ratio)
        
        record = Record(app_id, cate_index_real,
            cate_index_pred, conf_max, magnitude_ratio)
        records.append(record)
    save_pickle(records, 'gradcam_color_magnitude.obj')
    
        



