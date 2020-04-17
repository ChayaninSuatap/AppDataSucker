from tensorflow.keras.models import load_model, Model
import numpy as np
import os
from global_util import save_pickle, load_pickle

def compute_preds_sc(sc_names, model_path,
    sc_fd, use_feature_vector, show_output=False):

    model = load_model(model_path)
    #drop softmax layer
    if use_feature_vector:
        input_layer = None
        output_layer = None
        for layer in model.layers:
            if layer.name == 'input_1':
                input_layer = layer
            elif layer.name == 'my_model_flatten':
                output_layer = layer
                break
        model = Model(input_layer.input, output_layer.output)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

    preds = {}
    scs = []
    sc_names_pred = []

    for sc_name in sc_names:
        try:
            sc = icon_util.load_icon_by_fn(sc_fd + sc_name, 256, 160, rotate_for_sc = True)
        except:
            continue
        
        if show_output: print(sc_name)

        scs.append(sc)
        sc_names_pred.append(sc_name)
        if len(scs) == 64:
            pred = model.predict(np.array(scs) / 255)
            for p, sc_name in zip(pred, sc_names_pred):
                preds[sc_name] = p
            scs = []
            sc_names_pred = []

    if len(scs) > 0:
        pred = model.predict(np.array(scs) / 255)
        for p, sc_name in zip(pred, sc_names_pred):
            preds[sc_name] = p

    return preds

if __name__ == '__main__':
    # sc_fd = 'screenshots.256.distincted.rem.human/'
    # sc_fns = list(os.listdir(sc_fd))
    # compute_preds_sc(sc_fns, )
    pass
