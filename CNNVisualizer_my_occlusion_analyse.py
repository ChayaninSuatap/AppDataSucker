import os
from icon_util import load_icon_by_fn, rgb_to_gray
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    fd = 'visualize_cnn/my_occlusion/word/'
    imgs = []
    for fn in os.listdir(fd):
        img = load_icon_by_fn(fd + fn, 128, 128) / 255 
        imgs.append(img)
    
    result = np.zeros((128,128,3))
    for i in range(0, 128, 16):
        for j in range(0, 128, 16):
            count = 0
            for img in imgs:
                if img[i:i+16,j:j+16,:].sum() > 0:
                    count += 1
                    result[i:i+16,j:j+16,:] += img[i:i+16,j:j+16,:]
                else:
                    print('fucked')
            result[i:i+16,j:j+16,:] /= count
    
    red = np.zeros_like(result)
    green = np.array(red)
    blue = np.array(red)
    red[:,:,0] = result[:,:,0]
    green[:,:,1] = result[:,:,1]
    blue[:,:,2] = result[:,:,2]
    
    plt.imshow(result)
    plt.show()

    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    model = load_model(model_path)

    pred = model.predict(np.array([result]))
    pred_index = np.argmax(pred[0])
    print(pred_index, pred[0][pred_index])





