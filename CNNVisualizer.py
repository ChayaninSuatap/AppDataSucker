from global_util import load_pickle, save_pickle
from icon_util import load_icon_by_fn
from tensorflow.keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
import keras_util

class CNNVisualizer:

    def __init__(self, model_path):
        self.model = load_model(model_path)

    def visualize_filter(self): #now only plot 6 x 3 feature
        for layer in self.model.layers:
            if 'conv' not in layer.name:
                continue
            filters, biases = layer.get_weights()
            print(layer.name, filters.shape)
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)

            # plot first few filters
            n_filters, ix = 6, 1
            for i in range(n_filters):
                # get the filter
                f = filters[:, :, :, i]
                # plot each channel separately
                for j in range(3):
                    # specify subplot and turn of axis
                    ax = plt.subplot(n_filters, 3, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(f[:, :, j], cmap='gray')
                    ix += 1
            # show the figure
            plt.show()

    def visualize_feature(self, layer_i, img_path):
        for layer in self.model.layers:
            if 'conv' not in layer.name: continue
            m = Model(inputs=self.model.inputs, outputs=layer.output)
            img = load_icon_by_fn(img_path, 128, 128)
            img = img/255
            feature_maps = m.predict(np.array([img]))
            print('output shape', feature_maps.shape)

            # plot all 64 maps in an 8x8 squares
            square = 8
            ix = 1
            for _ in range(square):
                for _ in range(square):
                    # specify subplot and turn of axis
                    ax = plt.subplot(square, square, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # plot filter channel in grayscale
                    plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                    ix += 1
            # show the figure
            plt.show()


if __name__ == '__main__':
    model_path = 'sim_search_t/models/icon_model2.4_k3_t-ep-433-loss-0.319-acc-0.898-vloss-3.493-vacc-0.380.hdf5'
    # img_path = ''
    v = CNNVisualizer(model_path)
    v.visualize_feature(layer_i = 1, img_path = 'icons.combine.recrawled/com.solitairemaker.solitaire.png')

    # aial = load_pickle('aial_seed_327.obj')
    # m = load_model(model_path)
    # aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=3, n_splits=4)
    # app_id_pred_d = []
    # for x in aial_test:
    #     app_id = x[0]
    #     print(app_id)
    #     img_path = 'icons.combine.recrawled/' + app_id + '.png'
    #     normed_img = load_icon_by_fn(img_path, 128, 128)/255
    #     pred = m.predict(np.array([normed_img]))
    #     conf = max(pred[0])
    #     app_id_pred_d.append( (app_id,conf))
    # sorted_preds = sorted(app_id_pred_d, key = lambda x : x[1], reverse=True)
    # print(sorted_preds[:10])
    # save_pickle(sorted_preds, 'sorted_preds_icon_model2.4_k3.obj')

    # obj = load_pickle('sorted_preds_icon_model2.4_k3.obj')
    # count = 0
    # for app_id, conf in obj:
    #     if conf == 1.0: count+=1
    # print(count/len(obj))

    


        
    



