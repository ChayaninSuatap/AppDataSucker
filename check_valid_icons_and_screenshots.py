import sc_util
import mypath
import db_util
import icon_util
import preprocess_util
from ensemble_model_util import argmax
mypath.screenshot_folder = 'screenshots.256.distincted/'
mypath.icon_folder = 'icon.combine.recrawled/'
sc_dict = sc_util.make_sc_dict()

aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)
print(len(sc_dict))
input()
icon_each_cate = [0] * 17
sc_each_cate = [0] * 17
for app_id,_,cate_onehot in aial:
    try:
        icon = icon_util.load_icon_by_app_id(app_id,128,128)
        if app_id in sc_dict:
            found_sc=0
            for sc_fn in sc_dict[app_id]:
                try:
                    sc = icon_util.load_icon_by_fn('E:/thesis_datasets/screenshots.256.distincted/' + sc_fn, 256, 160, rotate_for_sc=True)
                    found_sc+=1
                except:
                    pass

            if found_sc > 0:        
                cate = argmax(cate_onehot)
                icon_each_cate[cate] += 1
                sc_each_cate[cate] += found_sc
        print(icon_each_cate)
        print(sc_each_cate) 
    except:
        pass
print(icon_each_cate)
print(sc_each_cate)