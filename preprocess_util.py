import db_util
import numpy as np
import mypath
from overall_feature_util import _extract_category
import icon_util
from global_util import save_pickle, load_pickle
import os.path
import os
import sc_util
from shutil import copyfile

def prep_rating_category():
    conn = db_util.connect_db()
    app_ids_and_labels = []
    dat=conn.execute('select app_id, rating, category from app_data')
    for x in dat:
        if x[1] != None and x[2] != None:
            app_ids_and_labels.append( (x[0], x[1], x[2])) 

    #rating str to float
    for i in range(len(app_ids_and_labels)):
        app_id , rating, cate = app_ids_and_labels[i]
        rating = float(app_ids_and_labels[i][1])
        app_ids_and_labels[i] = app_id, rating, cate
    
    #prep category
    for i,x in enumerate(app_ids_and_labels):
        onehot = _extract_category(x[2])
        app_ids_and_labels[i] = x[0], x[1], onehot
    
    return app_ids_and_labels

def prep_rating_category_scamount_download(conn=None, for_softmax=False):
    if conn is None:
        conn = db_util.connect_db()
    app_ids_and_labels = []
    dat=conn.execute('select app_id, rating, category, screenshots_amount, download_amount, rating_amount  from app_data order by app_id')
    for x in dat:
        if x[1] != None and x[2] != None and x[3] != None and x[4] != None and x[5] != None:
            app_ids_and_labels.append( (x[0], x[1], x[2], x[3], x[4], x[5])) 

    #rating str to float
    output = []
    for i in range(len(app_ids_and_labels)):
        app_id , rating, cate, scamount, download, rating_amount = app_ids_and_labels[i]
        rating = float(rating)
        scamount = float(scamount)
        #download 
        download = download.replace(',','').replace('+','')
        download = int(download)
        #rating amount
        rating_amount = rating_amount.replace(',','')
        rating_amount = int(rating_amount)
        #category
        cate = _extract_category(cate)
        if all(y==0 for y in cate):
            continue
        output.append((app_id, rating, cate, scamount, download, rating_amount))
    
    return output

def get_app_id_rating_cate_from_aial(aial):
    newaial = []
    for x in aial:
        newaial.append( (x[0], x[1], x[2]))
    return newaial

def remove_low_rating_amount(aial, threshold):
    newaial = []
    for x in aial:
        if x[5] > threshold:
            newaial.append(x)
    return newaial

def get_app_ids_without_icon(save_obj=False):
    app_ids = db_util.get_all_app_id(db_util.connect_db())
    app_ids_without_icon = []
    for app_id in app_ids:
        try:
            icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
        except:
            if not os.path.isfile('icons.recrawled/' + app_id + '.png'):
                print(app_id)
                app_ids_without_icon.append( app_id)
    if save_obj: save_pickle(app_ids_without_icon, 'app_ids_without_icon.recrawled.obj')
    return app_ids_without_icon

def check_app_ids_for_human_test_obj_is_latest():
    human_icons_app_ids = load_pickle('app_ids_for_human_test.obj')
    human_icons_app_ids = [x[0] for x in human_icons_app_ids]
    print(human_icons_app_ids)
    from PIL import Image
    import os
    import numpy as np

    pickle_images = []
    human_test_images = []

    __background512 = Image.new('RGBA', (512, 512), (255,255,255))

    for app_id in human_icons_app_ids:
        img_fn = app_id + '.png'
        png = Image.open('icons.512/'+img_fn).convert('RGBA')
        img = Image.alpha_composite(__background512, png)
        img = np.array(img)[:,:,:3]
        pickle_images.append(img)
    
    for img_fn in os.listdir('icons_human_test'):
        png = Image.open('icons_human_test/'+img_fn).convert('RGBA')
        img = Image.alpha_composite(__background512, png)
        img = np.array(img)[:,:,:3]
        human_test_images.append(img)

    not_found_n = 0
    
    for pickle_img in pickle_images:
        for human_img in human_test_images:
            if np.sum(pickle_img - human_img) == 0:
                break
        else:
            not_found_n += 1


    print('ok', not_found_n)

def remove_human_icons_from_dir(fd_path):
    human_icons_app_ids = load_pickle('app_ids_for_human_test.obj')
    human_icons_app_ids = [x[0] for x in human_icons_app_ids]

    not_removed_n = 0

    for app_id in human_icons_app_ids:
        img_fn = app_id + '.png'
        try:
            os.remove(fd_path + '/' + img_fn)
        except:
            not_removed_n += 1
    
    print('not removed ', not_removed_n)

def remove_human_scs_from_dir(fd_path):

    human_icons_app_ids = load_pickle('app_ids_for_human_test.obj')
    human_icons_app_ids = [x[0] for x in human_icons_app_ids]

    del_app_n = 0

    for human_sc_i, app_id in enumerate(human_icons_app_ids):

        # human_sc = icon_util.load_icon_by_fn('screenshots_human_test/%d.png' % (human_sc_i,) , 256, 160, rotate_for_sc=True)

        #search for exist file
        exist_files = []
        for i in range(21):
            sc_fn = '%s%2d.png' % (app_id,i)
            if os.path.exists(fd_path + sc_fn):
                exist_files.append(fd_path + sc_fn)
            
        found = False
        
        for path in exist_files:
            try:
                os.remove(path)
                found = True
                print('remove', path)
            except:
                pass
        
        if found : del_app_n += 1
    
    print('del app n', del_app_n)

def filter_sc_fns_from_icon_fns_by_app_id(icon_dir, sc_dir):
    results = []
    sc_dict = sc_util.make_sc_dict(sc_dir)
    for icon_fn in os.listdir(icon_dir):
        app_id = icon_fn[:-4]
        if app_id in sc_dict:
            results += sc_dict[app_id]
    return results

def copy_sc_fns_to_dir(sc_fns, source_dir, dest_dir):
    for sc_fn in sc_fns:
        print('copying', sc_fn)
        copyfile(source_dir + sc_fn, dest_dir + sc_fn)


if __name__ == '__main__':
    
    # remove_human_icons_from_dir('similarity_search/icons_rem_dup_recrawl/')
    # remove_human_scs_from_dir('screenshots.256.distincted/')



    x = filter_sc_fns_from_icon_fns_by_app_id('similarity_search/icons_rem_dup_human_recrawl/', 'screenshots.256.distincted.rem.human/')
    print(len(x))
    copy_sc_fns_to_dir(x, 'e:/screenshots.distincted.rem.human/', 'e:/screenshots.t/')
    # o = get_app_ids_without_icon(save_obj=True)
    # print(o, len(o))
    # app_ids_without_icon = load_pickle('app_ids_without_icon.recrawled.obj')
    # import sc_util
    # sc_dict = sc_util.make_sc_dict()
    # has_sc = []
    # for x in app_ids_without_icon:
    #     if x in sc_dict and len(sc_dict[x]) > 0:
    #         print(x)
    #         has_sc.append(x)
    # print(has_sc, len(has_sc))
    

