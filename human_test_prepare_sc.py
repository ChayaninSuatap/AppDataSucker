import global_util
import sc_util
import random
import shutil

sc_dict = sc_util.make_sc_dict()
o = global_util.load_pickle('app_ids_for_human_test.obj')
for i,(app_id,label) in enumerate(o):
    scs_of_app_id = sc_dict[app_id]
    #pick random a screenshot from an app
    sc_fn = scs_of_app_id[ random.randint(0, len(scs_of_app_id)-1)]
    shutil.copyfile('screenshots/' + sc_fn, 'screenshots_human_test/' + str(i) + '.png')
    '''
    # fix 66 and 258 not readable
    if i==66 or i==258:
        sc_fn = scs_of_app_id[ random.randint(0, len(scs_of_app_id)-1)]
        shutil.copyfile('screenshots/' + sc_fn, 'screenshots_human_test/' + str(i) + '.png')
        print(app_id)
    '''

