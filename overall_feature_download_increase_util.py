import db_util
import random
from download_increase_util import _prepare_dataset
import global_util
import overall_feature_util
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Dropout, LeakyReLU, concatenate
import keras_util
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import  StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint

def make_app_id_overall_feature_d(prepare_dataset_data, old_db_path , use_rating_amount=True):
    app_ids = [x[0] for x in prepare_dataset_data]
    dat = _prepare_overall_feature_dataset(prepare_dataset_data, old_db_path, use_rating_amount=use_rating_amount)
    output = {}
    if len(app_ids) != len(dat):
        print(len(app_ids), len(dat))
        raise ValueError('data len inconsistent for zip')
    for  app_id, overall_feature_and_label in zip(app_ids,dat):
        output[app_id] = overall_feature_and_label
    return output

def prepare_overall_feature_dataset( old_db_path , new_db_path, random_seed, k_iter, use_rating_amount=True):
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    download_increase_db = _prepare_dataset(app_ids_d, old_db_path, new_db_path)
    random.seed(5)
    random.shuffle(download_increase_db)
    dat = _prepare_overall_feature_dataset(download_increase_db, old_db_path, use_rating_amount=use_rating_amount, get_app_id=True)

    (train_data , test_data) = keras_util.gen_k_fold_pass(dat, kf_pass = k_iter, n_splits=4)

    app_id_train = []
    cate_train = []
    sdk_version_train = []
    content_rating_train = []
    other_train = []
    label_train = [] 
    for  app_id, cate, sdk_version, content_rating, other, label in train_data:
        app_id_train.append(app_id)
        cate_train.append(cate)
        sdk_version_train.append(sdk_version)
        content_rating_train.append(content_rating)
        other_train.append(other)
        label_train.append(label)
    
    app_id_test = []
    cate_test = []
    sdk_version_test = []
    content_rating_test = []
    other_test = []
    label_test = []
    for app_id, cate, sdk_version, content_rating, other, label in test_data:
        app_id_test.append(app_id)
        cate_test.append(cate)
        sdk_version_test.append(sdk_version)
        content_rating_test.append(content_rating)
        other_test.append(other)
        label_test.append(label)
    
    output = {
        'app_id_train': app_id_train,
        'app_id_test': app_id_test,
        'train_cate':np.array(cate_train),
        'train_sdk_version': np.array(sdk_version_train),
        'train_content_rating' : np.array(content_rating_train),
        'train_others' : np.array(other_train),
        'train_labels' : np.array(label_train),
        'test_cate' : np.array(cate_test),
        'test_sdk_version' : np.array(sdk_version_test),
        'test_content_rating' : np.array(content_rating_test),
        'test_others' : np.array(other_test),
        'test_labels' : np.array(label_test)
    }

    return output

def _prepare_overall_feature_dataset(download_increase_db, old_db_path, use_rating_amount=True, get_app_id=False):
    '''download_increase_db = output from prepare_dataset()'''
    conn = db_util.connect_db(old_db_path)
    sql = """
    select app_id, rating, download_amount, category, price, rating_amount, app_version, last_update_date, sdk_version, in_app_products, screenshots_amount, content_rating,
    video_screenshot from app_data where not app_id like "%&%" and not rating is NULL"""
    app_id_label_d = {app_id:label for app_id,label in download_increase_db}

    dat = conn.execute(sql)
    cates = []
    sdk_versions = []
    content_ratings = []
    others = []
    labels = []

    db_d = {}
    for rec in dat:
        app_id = rec[0]
        (cate, sdk_version, content_rating), single_node_output_vec, _ = overall_feature_util.extract_feature_vec(
            rec[1:],
            use_download_amount=True,
            use_rating_amount=use_rating_amount
        )
        db_d[app_id] = (cate, sdk_version, content_rating, single_node_output_vec)
    
    output = []
    for app_id, label in app_id_label_d.items():
        if app_id in db_d:
            cate, sdk_version, content_rating, single_node_output_vec = db_d[app_id]
            t = (np.array(cate), np.array(sdk_version), np.array(content_rating), np.array(single_node_output_vec), np.array(label))
            if get_app_id:
                t = (app_id,) + t
            output.append(t)

    return output

def make_model(cate_nodes_size, sdk_version_nodes_size, content_rating_nodes_size, other_input_nodes_size, min_input_size=3, denses=[8,4]):
    #cate input
    cate_input = Input(shape=(cate_nodes_size,))
    sdk_version_input = Input(shape=(sdk_version_nodes_size,))
    content_rating_input = Input(shape=(content_rating_nodes_size,))
    other_input = Input(shape=(other_input_nodes_size,))

    def add_dense(input_layer, dense_size, dropout_rate=0.5):
        x = Dense(dense_size)(input_layer)
        x = Dropout(dropout_rate)(x)
        x = LeakyReLU()(x)
        output_layer = BatchNormalization()(x)
        return output_layer

    embed_cate = Dense(min_input_size)(cate_input)
    embed_sdk_version = Dense(min_input_size)(sdk_version_input)
    embed_content_rating = Dense(min_input_size)(content_rating_input)
    embed_other = Dense(min_input_size)(other_input)

    #merge inputs
    merged_inputs = concatenate([embed_cate, embed_sdk_version, embed_content_rating, embed_other])
    #insert denses
    x = add_dense(merged_inputs, denses[0])
    for dense_value in denses[1:]:
        x = add_dense(x, dense_value)
    output_layer = Dense(2, activation='softmax')(x)

    inputs = {
        'cate_input' : cate_input,
        'sdk_version_input': sdk_version_input,
        'content_rating_input':content_rating_input,
        'other_input': other_input}
    
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    return model

def best_val_acc(history, acc_greater_val_acc=False):
    idx = None
    maxv = None
    for i,x in enumerate(history.history['val_acc']):
        if maxv is None or x > maxv:
            if acc_greater_val_acc and history.history['acc'][i] < x: continue
            maxv = x
            idx = i

    return history.history['val_acc'][idx], idx

def best_val_get_f1(history):
    idx = None
    maxv = None
    for i,x in enumerate(history.history['val_get_f1']):
        if maxv is None or x > maxv:
            maxv = x
            idx = i

    return history.history['val_acc'][idx], idx

def oversampling(train_x, train_y):
    train_x = train_x.tolist()
    train_y = train_y.tolist()
    c0, c1 = 0, 0
    for x in train_y:
        if x[0] == 1: c0+=1
        else: c1+=1
    
    need_more_n = c0 - c1

    inc_x = []
    inc_y = []
    for x,y in zip(train_x, train_y):
        if y[0] == 0:
            inc_x.append(x)
            inc_y.append(y)

    for _ in range(need_more_n):
        idx = random.randint(0, len(inc_x)-1)
        train_x.append(inc_x[idx])
        train_y.append(inc_y[idx])
    
    return (train_x, np.array(train_y))

if __name__ == '__main__':
    confmats = []
    best_accs = []
    best_eps = []
    epochs = 30
    batch_size = 32
    scalers = []

    for k_iter in range(4):
        dataset = prepare_overall_feature_dataset(
            'crawl_data/first_version/data.db',
            'crawl_data/update_first_version_2020_09_12/data.db',
            random_seed=5,
            k_iter=k_iter,
            use_rating_amount=True)
        
        app_id_train = dataset['app_id_train']
        train_cate = dataset['train_cate']
        train_sdk_version = dataset['train_sdk_version']
        train_content_rating = dataset['train_content_rating']
        train_others = dataset['train_others']
        train_labels = dataset['train_labels']
        app_id_test = dataset['app_id_test']
        test_cate = dataset['test_cate']
        test_sdk_version = dataset['test_sdk_version']
        test_content_rating = dataset['test_content_rating']
        test_others = dataset['test_others']
        test_labels = dataset['test_labels']

        # train_set, train_labels = oversampling((train_cate, train_sdk_version, train_content_rating, train_others), train_labels)
        # train_cate = np.array(train_set[0])
        # train_sdk_version = np.array(train_set[1])
        # train_content_rating = np.array(train_set[2])
        # train_others = np.array(train_set[3])

        #check set distribution
        c0, c1 = 0, 0
        for x in test_labels:
            if x[0] == 1: c0+=1
            else: c1+=1
        # print('testset dist',c0, c1)

        #normalize
        scaler = StandardScaler()
        scaled = scaler.fit(train_others)
        train_others = scaler.transform(train_others)
        test_others = scaler.transform(test_others)
        scalers.append(scaler)

        model = make_model(
            len(train_cate[0]),
            len(train_sdk_version[0]),
            len(train_content_rating[0]),
            len(train_others[0]),
            min_input_size=4, denses=[8,4])

        #class weight
        y_ints = np.argmax(train_labels, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_ints), y_ints)
        cw = dict(enumerate(class_weights))
        print(cw)

        cp_best_ep = ModelCheckpoint('best_overall_di_k%.hdf5' % (k_iter,), monitor='val_acc', save_best_only=True, verbose=0, period=1)
        history = model.fit(x={
            'cate_input': train_cate,
            'sdk_version_input': train_sdk_version,
            'content_rating_input': train_content_rating,
            'other_input': train_others
        },y=train_labels, epochs=epochs,
        validation_data=({
            'cate_input': test_cate,
            'sdk_version_input': test_sdk_version,
            'content_rating_input': test_content_rating,
            'other_input': test_others}
        , test_labels),
        batch_size=batch_size, class_weight=cw,
            callbacks=[cp_best_ep],
            verbose=0)

        
        top_val_acc, top_val_ep = best_val_acc(history)
        print('%.3f %d' % (top_val_acc, top_val_ep))
        best_accs.append(top_val_acc)
        best_eps.append(top_val_ep)

        model = load_model('best_overall_di_k%.hdf5' % (k_iter,))

        #confusion matrix
        preds=model.predict({
            'cate_input': test_cate,
            'sdk_version_input': test_sdk_version,
            'content_rating_input': test_content_rating,
            'other_input': test_others})

        #save pred_d
        pred_d = {}
        for app_id, pred in zip(app_id_test, preds):
            pred_d[app_id] = pred
        global_util.save_pickle(pred_d, 'best_overall_di_k%d.obj' % (k_iter,))

        #create y_true
        y_true = []
        for label in test_labels:
            y_true.append(label[1])
        #create y_pred
        y_pred = []
        for pred in preds:
            y_pred.append(0 if pred[0] > pred[1] else 1)

        #show conf mat
        confmat = confusion_matrix(y_true, y_pred)
        print(confmat[0][0], confmat[0][1])
        print(confmat[1][0], confmat[1][1])
        
        confmats.append(confmat)
    
    print('final result')
    for acc, ep in zip(best_accs, best_eps):
        print('%.3f %d' % (acc, ep))
    final_confmats = sum(confmats)/4
    print(final_confmats[0][0], final_confmats[0][1])
    print(final_confmats[1][0], final_confmats[1][1])

    prec = final_confmats[1][1]/(final_confmats[0][1] + final_confmats[1][1])
    recall = final_confmats[1][1]/(final_confmats[1][1] + final_confmats[1][0])
    f1 = (prec*recall*2)/(prec+recall)
    print('%.3f' % (f1,))

    global_util.save_pickle(scalers, 'overall_others_scalers.obj')