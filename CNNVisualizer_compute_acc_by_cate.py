from global_util import load_pickle
from joblib import load
from icon_util import load_icon_by_fn
from keras_util import gen_k_fold_pass
import numpy as np
from tensorflow.keras.models import load_model, Model

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

if __name__ == '__main__':
    k_fold = 3
    icon_fd = 'icons.combine.recrawled/'
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'


    aial = load_pickle('aial_seed_327.obj')
    aial_filtered = [(app_id, cate) for app_id, _, cate, *_ in aial]
    _, aial_test = gen_k_fold_pass(aial_filtered, k_fold, 4)

    aial_test_by_cate = {}
    for app_id, cate in aial_test:
        cate_index = np.array(cate).argmax()
        if cate_index not in aial_test_by_cate:
            aial_test_by_cate[cate_index] = [],[]
        img = load_icon_by_fn(icon_fd + app_id + '.png', 128, 128)/255
        aial_test_by_cate[cate_index][0].append(img)
        aial_test_by_cate[cate_index][1].append(cate)

    model = load_model(model_path)
    accs = []
    for cate_index in range(17):
        print('evaluating', cates[cate_index])
        loss, acc = model.evaluate(np.array(aial_test_by_cate[cate_index][0]), np.array(aial_test_by_cate[cate_index][1]))
        accs.append((cates[cate_index], acc))
    
    accs_sorted = sorted(accs, key = lambda x: x[1], reverse=True)
    print(accs_sorted)

