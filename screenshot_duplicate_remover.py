import icon_util
import os
import scipy.misc
import sc_util

THRESHOLD = 1000

sc_dict = sc_util.make_sc_dict()

for k,sc_fn_s in sc_dict.items():
    # loop each app_Id
    app_id_scs = []
    for sc_fn in sc_fn_s:
        path = 'screenshots/' + sc_fn
        try:
            sc = icon_util.load_icon_by_fn(path, 256, 160, rotate_for_sc=True)
            app_id_scs.append(  (sc, sc_fn) )
        except:
            continue
    #save only not duplicate
    for i in range(len(app_id_scs)):
        #loop check duplicate
        duplicate_flag = False
        for j in range(i+1, len(app_id_scs)):
            subed = app_id_scs[i][0] - app_id_scs[j][0]
            if subed.sum() < THRESHOLD:
                duplicate_flag = True
                break
        if duplicate_flag == False:
            print('save', app_id_scs[i][1])
            scipy.misc.imsave('screenshots.256.distincted/' + app_id_scs[i][1], app_id_scs[i][0])
