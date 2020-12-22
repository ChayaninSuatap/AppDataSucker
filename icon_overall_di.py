import overall_feature_download_increase_util as overall
import download_increase_util as di
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, concatenate, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import linear
from keras.utils.vis_utils import plot_model
import global_util
import random
import keras_util
from image_batch_generator import image_batch_sequence
import numpy as np
from sklearn.preprocessing import  StandardScaler


def make_icon_overall_model(icon_model, dropout_rate=0.5, overall_make_model_param={}, use_pretrained_model=False):
    overall_model = overall.make_model(**overall_make_model_param)
    # rename overall layers
    for layer in overall_model.layers:
        layer._name = 'overall_'+layer.name
    overall_model.compile(loss='categorical_crossentropy', metrics=['acc'])

    if not use_pretrained_model:
        icon_model = di.extend_cate_model(icon_model)

    # drop last layer of overall and icon
    overall_model.layers[-1].activation = linear
    icon_model.layers[-1].activation = linear
    last_layer_overall = overall_model.layers[-1].output
    last_layer_icon = icon_model.layers[-1].output

    concat = concatenate([last_layer_overall, last_layer_icon])
    x = Dropout(dropout_rate,  name='icon_overall_dropout')(concat)
    x = LeakyReLU(name='icon_overall_leakyrelu')(x)
    x = BatchNormalization(name='icon_overall_batchnormalization')(x)
    output_layer = Dense(2, activation='softmax',
                         name='icon_overall_output')(x)

    new_model = Model(inputs=[
        icon_model.layers[0].input,
        overall_model.layers[0].input,
        overall_model.layers[1].input,
        overall_model.layers[2].input,
        overall_model.layers[3].input,
    ],
        outputs=output_layer
    )
    new_model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['acc'])
    return new_model

def make_sc_overall_other_scalers_obj():
    
    data = global_util.load_pickle('sc_di_aial.obj')
    preped_dat = global_util.load_pickle('preped_dat.obj')
    app_id_overall_feature_d = overall.make_app_id_overall_feature_d(
        preped_dat,
        'crawl_data/first_version/data.db')
    
    scalers = []

    for i in range(4):
        train_data, _ = keras_util.gen_k_fold_pass(
            data, kf_pass=i, n_splits=4)

        other_features = []
        for app_id, _ in train_data:
            overall_feature = app_id_overall_feature_d[app_id[:-6]]
            other_feature = overall_feature[-2]
            other_features.append(other_feature)
        scaler = StandardScaler()
        scaler.fit(np.array(other_features))
        scalers.append( scaler)
    
    global_util.save_pickle(scalers, 'sc_overall_other_scalers.obj')

def make_sc_overall_other_scalers_obj_split_by_icon_split():
    preped_dat = global_util.load_pickle('preped_dat.obj')
    app_id_overall_feature_d = overall.make_app_id_overall_feature_d(
        preped_dat,
        'crawl_data/first_version/data.db')
    
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]: True for x in aial}

    scalers = []

    for i in range(4):
        data_icon = di._prepare_dataset(
            app_ids_d,
            'crawl_data/first_version/data.db',
            'crawl_data/update_first_version_2020_09_12/data.db')
        random.seed(5)
        random.shuffle(data_icon)
        train_data_icon, test_data_icon = keras_util.gen_k_fold_pass(
            data_icon, kf_pass=i, n_splits=4)
        
        sc_dat = global_util.load_pickle('sc_di_aial.obj')
        train_data, test_data = split_sc_by_icon_split(train_data_icon, test_data_icon, sc_dat)

        other_features = []
        for app_id, _ in train_data:
            overall_feature = app_id_overall_feature_d[app_id[:-6]]
            other_feature = overall_feature[-2]
            other_features.append(other_feature)
        scaler = StandardScaler()
        scaler.fit(np.array(other_features))
        scalers.append( scaler)
    
    global_util.save_pickle(scalers, 'sc_overall_other_scalers_split_by_icon_split.obj')

def split_sc_by_icon_split(icon_train_pairs, icon_test_pairs, sc_dat):
    icon_train_dat = {}
    icon_test_dat = {}

    for app_id,_ in icon_train_pairs:
        icon_train_dat[app_id] = True
    for app_id,_ in icon_test_pairs:
        icon_test_dat[app_id] = True
    
    sc_train, sc_test = [], []

    for app_id, label in sc_dat:
        if app_id[:-6] in icon_train_dat:
            sc_train.append((app_id, label))
        elif app_id[:-6] in icon_test_dat:
            sc_test.append((app_id, label))
        else:
            raise ValueError('no match app_id in both train and test')
    
    return sc_train, sc_test

if __name__ == '__main__':
    # make_sc_overall_other_scalers_obj_split_by_icon_split()
    # input('complete')
    overall_make_model_param = {
        'cate_nodes_size': 17, 'sdk_version_nodes_size': 38, 'content_rating_nodes_size': 38, 'other_input_nodes_size' : 9}

    # icon
    # model = make_icon_overall_model(
    #     load_model('sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5'),
    #     overall_make_model_param=overall_make_model_param
    # )

    # sc
    model = make_icon_overall_model(
        load_model('C:/Users/chaya/Downloads/model-ep-085-loss-0.099-acc-0.965-vloss-1.180-vacc-0.742.hdf5'),
        overall_make_model_param=overall_make_model_param,
        use_pretrained_model = True
    )

    for i, w in enumerate(model.weights):
        split_name = w.name.split('/')
        new_name = split_name[0] + '_' + str(i) + '/' + split_name[1] + '_' + str(i)
        model.weights[i]._handle_name = new_name

    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]: True for x in aial}

    #icon
    data_icon = di._prepare_dataset(
        app_ids_d,
        'crawl_data/first_version/data.db',
        'crawl_data/update_first_version_2020_09_12/data.db')
    random.seed(5)
    random.shuffle(data_icon)
    train_data_icon, test_data_icon = keras_util.gen_k_fold_pass(
        data_icon, kf_pass=3, n_splits=4)

    # #sc
    # data, preped_data = di._prepare_dataset_sc(
    #     'screenshots.256.distincted.rem.human',
    #     app_ids_d=app_ids_d,
    #     old_db_path='crawl_data/first_version/data.db',
    #     new_db_path='crawl_data/update_first_version_2020_09_12/data.db')
    # random.seed(5)
    # random.shuffle(data)
    sc_dat = global_util.load_pickle('sc_di_aial.obj')
    train_data, test_data = split_sc_by_icon_split(train_data_icon, test_data_icon, sc_dat)
    print(len(train_data), len(test_data))

    c0, c1 = 0, 0
    for x in test_data:
        if x[-1][0] == 1: c0+=1
        else: c1+=1
    print(c0, c1)

    ad = {}
    bd = {}

    for k,v in train_data:
        ad[k[:-6]] = True
    for k,v in test_data:
        bd[k[:-6]] = True
    for k in ad.keys():
        if k in bd.keys():
            print(k)
    input('not dup ? maybe ?')

    # train_data, test_data = keras_util.gen_k_fold_pass(
        # data, kf_pass=0, n_splits=4)

    app_id_overall_feature_d = overall.make_app_id_overall_feature_d(
        global_util.load_pickle('preped_dat.obj'),
        'crawl_data/first_version/data.db')
    
    # class weight
    cw = di.compute_class_weight([x[1] for x in train_data])
    print('class weight', cw)
    # sequences
    # overall other scalers
    scalers = global_util.load_pickle('sc_overall_other_scalers.obj')

    #sc (oh forget to keep icon case)
    train_seq = image_batch_sequence(
        train_data[:8], batch_size=8, app_id_overall_feature_d=app_id_overall_feature_d,  overall_other_scaler=scalers[0],
        train_sc=True, sc_fd='screenshots.256.distincted.rem.human')
    test_seq = image_batch_sequence(
        test_data[:8], batch_size=8, shuffle=False, app_id_overall_feature_d=app_id_overall_feature_d, overall_other_scaler=scalers[0],
        train_sc=True, sc_fd='screenshots.256.distincted.rem.human')

    from tensorflow.keras.callbacks import ModelCheckpoint
    model.fit(train_seq, validation_data=test_seq, epochs=1, batch_size=8, callbacks=[ModelCheckpoint('t_{val_acc:.3f}.hdf5', monitor='val_acc', period=1, save_best_only=True)])
    
