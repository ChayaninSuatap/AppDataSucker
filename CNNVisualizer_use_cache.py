from global_util import load_pickle
from CNNVisualizer_with_cache import Record
import numpy as np

if __name__ == '__main__':
    cates = ['BOARD', 'TRIVIA',	'ARCADE','CARD','MUSIC','RACING','ACTION','PUZZLE','SIMULATION','STRATEGY','ROLE_PLAYING','SPORTS','ADVENTURE','CASINO','WORD','CASUAL','EDUCATIONAL']

    filter_cate = 'BOARD'
    filter_cate_index = cates.index(filter_cate)
    min_conf = 0.9

    o = load_pickle('gradcam_color_magnitude.obj')


    for filter_cate in cates:
        filter_cate_index = cates.index(filter_cate)
        ms = []
        count = 0
        print(filter_cate, end=' ')
        for e in o:
            if e.pred_conf >= min_conf and e.cate_index_real == filter_cate_index \
                and e.cate_index_pred == filter_cate_index:
                ms.append(e.magnitude_ratio)
                count += 1
        if len(ms)==0:
            ms.append(np.array([.0, .0, .0]))
        mean = np.array(ms).mean(axis=0)
        std = np.array(ms).mean(axis=0)
        print('%f %f %f %f %f %f %d' % (mean[0], mean[1], mean[2], std[0], std[1], std[2], count))
    
