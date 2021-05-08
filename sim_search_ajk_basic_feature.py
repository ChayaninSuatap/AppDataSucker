from joblib import dump, load
import os
import global_util

def make_filtered_app_id_basic_feature_d(feature_d_paths, walk_root_path, save_path, is_sc):
    output = {}
    app_ids = []
    for _, _, fns in os.walk(walk_root_path):
        for fn in fns:
            if not is_sc:
                app_id = fn[:-4]
            else:
                app_id = fn
            app_ids.append(app_id)

    for feature_d_path in feature_d_paths:
        feature_d = load(feature_d_path)
        for k,v in feature_d.items():
            if k in app_ids[:]:
                output[k] = v
                app_ids.remove(k)

    global_util.save_pickle(output, save_path) 
    print(app_ids)

if __name__ == '__main__':
    walk_root_path = 'journal/sim_search/sports/icons/'
    save_path = 'journal/sim_search/sports/hog16_icon_features.obj'
    feature_d_paths = [
        'basic_features_t/icon_hog16_t.gzip'
        # 'journal/basic_features/hog16_sc/hog16_sc_test_k0.gzip',
        # 'journal/basic_features/hog16_sc/hog16_sc_test_k1.gzip',
        # 'journal/basic_features/hog16_sc/hog16_sc_test_k2.gzip',
        # 'journal/basic_features/hog16_sc/hog16_sc_test_k3.gzip',
    ]
    is_sc=False
    make_filtered_app_id_basic_feature_d(feature_d_paths, walk_root_path, save_path, is_sc)