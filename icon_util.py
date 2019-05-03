import numpy as np
from PIL import Image
import mypath
import random

def load_icon_by_app_id(app_id, resizeW, resizeH):
    return open_and_resize(mypath.icon_folder + app_id + '.png', resizeW, resizeH)

def open_and_resize(fn, resizeW, resizeH):
    return np.asarray( _convert_to_rgba(fn, resizeW, resizeH ))[:,:,:3]

def _convert_to_rgba(fn, resizeW, resizeH):
    png = Image.open(fn).convert('RGBA')
    png = png.resize( (resizeW, resizeH))
    background = Image.new('RGBA', png.size, (255,255,255))

    alpha_composite = Image.alpha_composite(background, png)
    return alpha_composite

def oversample_image(app_ids_and_labels):
    app_id_pool = {}
    label_counter = {}
    for app_id, label in app_ids_and_labels:
        #add in pool
        if label not in app_id_pool: app_id_pool[label] = []
        app_id_pool[label].append((app_id,label))
        if label  not in label_counter: label_counter[label] = 0
        label_counter[label] += 1
    #start sampling
    max_freq = max( list(label_counter.values()))
    for label in label_counter.keys():
        for i in range(max_freq - label_counter[label]):
            #pick and app_id
            picked = random.choice(app_id_pool[label])
            app_ids_and_labels.append( picked)