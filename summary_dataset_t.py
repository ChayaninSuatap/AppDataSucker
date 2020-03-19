import preprocess_util
import icon_cate_util
import numpy as np
import random
import icon_util
import math
from tensorflow.keras.callbacks import ModelCheckpoint
from keras_util import PlotAccLossCallback, gen_k_fold_pass, metric_top_k, eval_top_5, compute_class_weight, SaveBestEpochCallback
from tensorflow.keras.models import load_model
import keras
import functools
import icon_cate_data_export
import global_util
import mypath
import keras_util
import matplotlib.pyplot as plt
import sc_util
from overall_feature_util import _all_game_category as cates

mypath.icon_folder = 'similarity_search/icons_rem_dup_human_recrawl/'
mypath.screenshot_folder = 'screenshots.256.distincted.rem.human/'
random.seed(327)
np.random.seed(327)
aial = icon_cate_util.make_aial_from_seed(327, mypath.icon_folder)
aial = icon_cate_util.filter_aial_rating_cate(aial)

sc_dict = sc_util.make_sc_dict(mypath.screenshot_folder)

icon_and_sc = {}

count = 0
for app_id,_,cate in aial:
    if app_id in sc_dict and len(sc_dict[app_id]) > 0:

        cate_index = np.array(cate).argmax()
        cate_str = cates[cate_index]
        if cate_str not in icon_and_sc:
            icon_and_sc[cate_str] = [0, 0]

        icon_and_sc[cate_str][0] += 1
        icon_and_sc[cate_str][1] += len(sc_dict[app_id])

icon_sc_tuple = []
for k,v in icon_and_sc.items():
    icon_sc_tuple.append( (k, v[0], v[1]))

sorted_icon_sc_tuple = sorted( icon_sc_tuple, key= lambda x:x[1], reverse=True)

for x in sorted_icon_sc_tuple:
    print(x)




