import db_util
import random
from download_increase_util import prepare_dataset
import global_util
import overall_feature_util
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Input, BatchNormalization, Dropout, LeakyReLU, concatenate

def prepare_overall_feature_dataset(download_increase_db, old_db_path, random_seed = 5):
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
    for rec in dat:
        app_id = rec[0]
        if app_id in app_id_label_d:
            [cate, sdk_version, content_rating], single_node_output_vec, _ = overall_feature_util.extract_feature_vec(rec[1:], use_download_amount=False)
            cates.append(cate)
            sdk_versions.append(sdk_version)
            content_ratings.append(content_rating)
            others.append(single_node_output_vec)
            labels.append(app_id_label_d[app_id])

    return np.array(cates), np.array(sdk_versions), np.array(content_ratings), np.array(others), np.array(labels)

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

if __name__ == '__main__':
    download_increase_db = prepare_dataset()
    overall_feature_db = prepare_overall_feature_dataset(download_increase_db, 'crawl_data/first_version/data.db')

    random.seed(5)
    random.shuffle(overall_feature_db)

    cates, sdk_versions, content_ratings, others, labels = overall_feature_db

    model = make_model(len(cates[0]), len(sdk_versions[0]), len(content_ratings[0]), len(others[0]))
    model.fit([cates, sdk_versions, content_ratings, others], labels, epochs=5)

