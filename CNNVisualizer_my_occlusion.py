
from icon_util import load_icon_by_fn, rgb_to_gray
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

def compute_occlusion(img, model, cate_index):

    #find best i j for bootstraping
    bootstrap_result = []
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            
            patched = np.zeros_like(img)
            patched[i:i+16,j:j+16,:] = img[i:i+16,j:j+16,:]

            pred = model.predict(np.array([patched]))
            bootstrap_result.append((pred[0][cate_index], patched))

    bs_patched = max(bootstrap_result, key = lambda x : x[0])    

    patched_now = np.array(bs_patched[1])
    conf_now = bs_patched[0]

    turned_on_cells = []
    pred_times = 0
    while True:
        conf_this_iter = conf_now
        for i in range(0, 128, 16):
            for j in range(0, 128, 16):

                if i*(128/16)+j in turned_on_cells:
                    continue
                
                patched = np.array(patched_now)
                patched[i:i+16,j:j+16,:] = img[i:i+16,j:j+16,:]
                pred = model.predict(np.array([patched]))
                pred_times += 1

                if pred[0][cate_index] > conf_now:
                    conf_now = pred[0][cate_index]
                    patched_now = patched 
                    turned_on_cells.append(i*(128/16)+j)
        if conf_this_iter == conf_now:
            break
    
    print('last pred', conf_now)
    return patched_now


            



if __name__ == '__main__':
    app_id = 'com.neuralplay.android.spades'
    img_path = 'icons.combine.recrawled/%s.png' % (app_id,)
    img = load_icon_by_fn(img_path, 128, 128)/255
    cate_index = 7
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    model = load_model(model_path)

    result = compute_occlusion(img, model, cate_index)
    plt.imshow(result)
    plt.show()