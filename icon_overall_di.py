import overall_feature_download_increase_util as overall
import download_increase_util as di
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, concatenate, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import linear
from keras.utils.vis_utils import plot_model

def make_icon_overall_model(icon_model_path, dropout_rate=0.5, overall_make_model_param={}):
    overall_model = overall.make_model(**overall_make_model_param)
    #rename overall layers
    for layer in overall_model.layers:
        layer._name = 'overall_'+layer.name
    overall_model.compile(loss='categorical_crossentropy', metrics=['acc'])

    icon_model = load_model(icon_model_path)
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
    output_layer = Dense(2, activation='softmax', name='icon_overall_output')(x)

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
        'cate_nodes_size': 17, 'sdk_version_nodes_size': 34, 'content_rating_nodes_size': 34}
    model = make_icon_overall_model(
        icon_model_path='sim_search_t/models/icon_model2.4_k0_t-ep-404-loss-0.318-acc-0.896-vloss-3.674-vacc-0.357.hdf5',
        overall_make_model_param=overall_make_model_param
    )
