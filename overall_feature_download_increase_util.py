import db_util
import random
from download_increase_util import _prepare_dataset
import global_util
import overall_feature_util
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Dropout, LeakyReLU, concatenate
import keras_util
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def prepare_overall_feature_dataset( old_db_path , new_db_path, random_seed, k_iter, use_rating_amount=True):
    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]:True for x in aial}
    download_increase_db = _prepare_dataset(app_ids_d, old_db_path, new_db_path)
    random.seed(5)
    random.shuffle(download_increase_db)
    dat = _prepare_overall_feature_dataset(download_increase_db, old_db_path, use_rating_amount=use_rating_amount)

    train_data , test_data = keras_util.gen_k_fold_pass(dat, kf_pass = k_iter, n_splits=4)

    cate_train = []
    sdk_version_train = []
    content_rating_train = []
    other_train = []
    label_train = [] 
    for cate, sdk_version, content_rating, other, label in train_data:
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
    
    return (np.array(cate_train), np.array(sdk_version_train), np.array(content_rating_train), np.array(other_train), np.array(label_train)), \
    (np.array(cate_test), np.array(sdk_version_test), np.array(content_rating_test), np.array(other_test), np.array(label_test))

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
        [cate, sdk_version, content_rating], single_node_output_vec, _ = overall_feature_util.extract_feature_vec(
            rec[1:],
            use_download_amount=False,
            use_rating_amount=use_rating_amount
        )
        db_d[app_id] = (cate, sdk_version, content_rating, single_node_output_vec)
    
    output = []
    for app_id, label in app_id_label_d.items():
        if app_id in db_d:
            cate, sdk_version, content_rating, single_node_output_vec = db_d[app_id]
            output.append((cate, sdk_version, content_rating, single_node_output_vec, label))

    return output

def make_model(cate_nodes_size, sdk_version_nodes_size, content_rating_nodes_size, other_input_nodes_size=8, min_input_size=3):
    #cate input
    cate_input = Input(shape=(cate_nodes_size,))
    sdk_version_input = Input(shape=(sdk_version_nodes_size,))
    content_rating_input = Input(shape=(content_rating_nodes_size, ))
    other_input = Input(shape=(other_input_nodes_size,))

    def add_dense(input_layer, dense_size, dropout_rate=0.5):
        x = Dropout(dropout_rate)(input_layer)
        x = Dense(dense_size)(x)
        x = LeakyReLU()(x)
        output_layer = BatchNormalization()(x)
        return output_layer

    #minimize inputs
    min_cate_input = add_dense(cate_input, min_input_size)
    min_sdk_version_input = add_dense(sdk_version_input, min_input_size)
    min_content_rating_input = add_dense(content_rating_input, min_input_size)

    #merge inputs
    x = concatenate([min_cate_input, min_sdk_version_input, min_content_rating_input, other_input])
    #insert denses
    x = add_dense(x, 16)
    x = add_dense(x, 8)
    x = add_dense(x, 4)
    output_layer = Dense(2, activation='softmax')(x)

    model = Model(inputs=[cate_input, sdk_version_input, content_rating_input, other_input], outputs=output_layer)
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    return model

def best_val_acc(history):
    idx = np.array(history.history['val_acc']).argmax()
    return history.history['val_acc'][idx], idx


if __name__ == '__main__':
    for k_iter in range(4):
        overall_feature_db = prepare_overall_feature_dataset(
            'crawl_data/first_version/data.db',
            'crawl_data/update_first_version_2020_09_12/data.db',
            random_seed=5,
            k_iter=k_iter,
            use_rating_amount=False)

        trainset, testset = overall_feature_db
        # c0, c1 = 0, 0
        # for x in trainset[-1]:
            # if x[0] == 1: c0+=1
            # else: c1+=1
        # print(c0, c1)

        cates, sdk_versions, content_ratings, others, _ = trainset

        model = make_model(len(cates[0]), len(sdk_versions[0]), len(content_ratings[0]), len(others[0]))

        #class weight
        y_ints = np.argmax(trainset[-1], axis=1)
        class_weights = compute_class_weight('balanced', np.unique(y_ints), y_ints)
        cw = dict(enumerate(class_weights))
        print(cw)

        history = model.fit(trainset[:-1], trainset[-1], epochs=10, validation_data=(testset[:-1], testset[-1]), verbose=0)
        print(best_val_acc(history))

        #confusion matrix
        preds=model.predict(testset)
        #create y_true
        y_true = []
        for label in testset[-1]:
            y_true.append(label[1])
        #create y_pred
        y_pred = []
        for pred in preds:
            y_pred.append(0 if pred[0] > pred[1] else 0)

        #show conf mat
        print(confusion_matrix(y_true, y_pred))

