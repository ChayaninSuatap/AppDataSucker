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


def prepare_overall_feature_dataset( old_db_path , new_db_path, random_seed, k_iter, use_rating_amount=True):
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    download_increase_db = _prepare_dataset(app_ids_d, old_db_path, new_db_path)
    random.seed(5)
    random.shuffle(download_increase_db)
    dat = _prepare_overall_feature_dataset(download_increase_db, old_db_path, use_rating_amount=use_rating_amount)

    (train_data , test_data) = keras_util.gen_k_fold_pass(dat, kf_pass = k_iter, n_splits=4)

    cate_train = []
    sdk_version_train = []
    content_rating_train = []
    other_train = []
    label_train = [] 
    for  cate, sdk_version, content_rating, other, label in train_data:
        cate_train.append(cate)
        sdk_version_train.append(sdk_version)
        content_rating_train.append(content_rating)
        other_train.append(other)
        label_train.append(label)
    
    cate_test = []
    sdk_version_test = []
    content_rating_test = []
    other_test = []
    label_test = []
    for cate, sdk_version, content_rating, other, label in test_data:
        cate_test.append(cate)
        sdk_version_test.append(sdk_version)
        content_rating_test.append(content_rating)
        other_test.append(other)
        label_test.append(label)
    
    output = {
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

def _prepare_overall_feature_dataset(download_increase_db, old_db_path, use_rating_amount=True):
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
            output.append((np.array(cate), np.array(sdk_version), np.array(content_rating), np.array(single_node_output_vec), np.array(label)))

    return output

def make_model(cate_nodes_size, sdk_version_nodes_size, content_rating_nodes_size, other_input_nodes_size=8, min_input_size=3):
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
    x = add_dense(merged_inputs, 16)
    x = add_dense(x, 8)
    x = add_dense(x, 4)
    output_layer = Dense(2, activation='softmax')(x)

    # model = Model(inputs=[cate_input, sdk_version_input, content_rating_input, other_input], outputs=output_layer)
    inputs = {
        'cate_input' : cate_input,
        'sdk_version_input': sdk_version_input,
        'content_rating_input':content_rating_input,
        'other_input': other_input}
    
    model = Model(inputs=inputs, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    return model

def best_val_acc(history):
    idx = None
    maxv = None
    for i,x in enumerate(history.history['val_acc']):
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
    
    return np.array(train_x), np.array(train_y)

if __name__ == '__main__':
    for k_iter in range(4):
        dataset = prepare_overall_feature_dataset(
            'crawl_data/first_version/data.db',
            'crawl_data/update_first_version_2020_09_12/data.db',
            random_seed=5,
            k_iter=k_iter,
            use_rating_amount=True)
        
        epochs = 1
        batch_size = 8

        train_cate = dataset['train_cate']
        train_sdk_version = dataset['train_sdk_version']
        train_content_rating = dataset['train_content_rating']
        train_others = dataset['train_others']
        train_labels = dataset['train_labels']
        test_cate = dataset['test_cate']
        test_sdk_version = dataset['test_sdk_version']
        test_content_rating = dataset['test_content_rating']
        test_others = dataset['test_others']
        test_labels = dataset['test_labels']

        # train_others, train_labels = oversampling(train_others, train_labels)

        #check set distribution
        c0, c1 = 0, 0
        for x in train_labels:
            if x[0] == 1: c0+=1
            else: c1+=1
        # print(c0, c1)

        #normalize
        scaler = StandardScaler()
        scaled = scaler.fit(train_others)
        train_others = scaler.transform(train_others)
        test_others = scaler.transform(test_others)

        model = make_model(
            len(train_cate[0]),
            len(train_sdk_version[0]),
            len(train_content_rating[0]),
            len(train_others[0]))
        model.summary()

        #class weight
        y_ints = np.argmax(train_labels, axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_ints), y_ints)
        cw = dict(enumerate(class_weights))
        print(cw)

        cp_best_ep = ModelCheckpoint('best_overall_download_increase.hdf5', monitor='val_acc', save_best_only=True, verbose=0, period=1)
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
            # callbacks=[cp_best_ep],
            verbose=1)
        top_val_acc, top_val_ep = best_val_acc(history)
        print('%.3f %d' % (top_val_acc, top_val_ep))

        model = load_model('best_overall_download_increase.hdf5')

        #confusion matrix
        preds=model.predict(x=[test_cate, test_sdk_version, test_content_rating, test_others])

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
