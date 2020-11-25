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


def make_icon_overall_model(icon_model, dropout_rate=0.5, overall_make_model_param={}):
    overall_model = overall.make_model(**overall_make_model_param)
    # rename overall layers
    for layer in overall_model.layers:
        layer._name = 'overall_'+layer.name
    overall_model.compile(loss='categorical_crossentropy', metrics=['acc'])

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


if __name__ == '__main__':
    overall_make_model_param = {
        'cate_nodes_size': 17, 'sdk_version_nodes_size': 38, 'content_rating_nodes_size': 38, 'other_input_nodes_size' : 9}
    model = make_icon_overall_model(
        load_model('sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5'),
        overall_make_model_param=overall_make_model_param
    )

    aial = global_util.load_pickle('aial_seed_327.obj')
    app_ids_d = {x[0]: True for x in aial}
    data = di._prepare_dataset(
        app_ids_d,
        'crawl_data/first_version/data.db',
        'crawl_data/update_first_version_2020_09_12/data.db')
    random.seed(5)
    random.shuffle(data)
    train_data, test_data = keras_util.gen_k_fold_pass(
        data, kf_pass=0, n_splits=4)
    app_id_overall_feature_d = overall.make_app_id_overall_feature_d(
        data,
        'crawl_data/first_version/data.db')
    # class weight
    cw = di.compute_class_weight([x[1] for x in train_data])
    print('class weight', cw)
    # sequences
    # overall other scalers
    scalers = global_util.load_pickle('overall_other_scalers.obj')
    train_seq = image_batch_sequence(
        train_data, batch_size=8, app_id_overall_feature_d=app_id_overall_feature_d)
    test_seq = image_batch_sequence(
        test_data, batch_size=8, shuffle=False, app_id_overall_feature_d=app_id_overall_feature_d, overall_other_scaler=scalers[0])

    model.fit(train_seq, epochs=1, batch_size=8)
