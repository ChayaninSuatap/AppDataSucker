
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from icon_util import load_icon_by_fn, shift_hue
from tf_explain_util import visualize_grad_cam

cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

if __name__ == '__main__':
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    icon_fn = 'visualize_cnn/gradcam/PUZZLE/handmade/rect_color_rot_no_red.png'
    m = load_model(model_path)

    normed_img = load_icon_by_fn(icon_fn, 128, 128)/255

    pred = m.predict(np.array([normed_img]))
    conf = max(pred[0])
        
    #show predict stats
    cate_index = 7#np.argmax(pred[0])
    pred_tuple = []
    for pred_val, cate_str in zip(pred[0], cates):
        pred_tuple.append( (cate_str,pred_val))
    pred_tuple = sorted(pred_tuple, key = lambda x: x[1], reverse=True)
    print(pred_tuple[:5])

    visualize_grad_cam(m, normed_img, cate_index, compute_color_magnitude=False)