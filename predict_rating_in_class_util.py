from icon_cate_util import create_icon_cate_model, filter_aial_rating_cate
from global_util import load_pickle
from keras_util import gen_k_fold_pass
import keras_util
import icon_cate_util
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
import math
import mypath

def _override_last_dense_layer(model, new_dense_size):
    input_layer = model.layers[0].input
    last_layer = model.layers[-2].output
    last_layer = Dense(new_dense_size, name='class_rating_dense', activation='softmax')(last_layer)
    return Model(inputs=[input_layer], outputs=[last_layer])

def compile_model_decorator(fn):
    def wrapper(*args, **kwargs):
        model = fn(*args, **kwargs)
        model.compile(optimizer='adam',
            loss='categorical_crossentropy', metrics=['acc'])
        return model
    return wrapper

@compile_model_decorator
def  create_class_rating_model(create_cate_model_args, class_n):
    model = create_icon_cate_model(**create_cate_model_args)
    return _override_last_dense_layer(model, class_n)

@compile_model_decorator
def create_class_rating_model_from_pretrained_model(pretrained_model, class_n):
    return _override_last_dense_layer(pretrained_model, class_n)
    
if __name__ == '__main__':
    split_period = [3.5 , 4, 4.5, 5]
    k_iter = 0
    batch_size=32
    epochs=50

    mypath.icon_folder = 'icons.combine.recrawled/'

    aial = load_pickle('aial_seed_327.obj')
    aial = filter_aial_rating_cate(aial)

    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass=k_iter, n_splits=4)
    gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs,
        enable_cache=True, datagen = keras_util.create_image_data_gen(),
        predict_class_rating=True, class_rating_split_period=split_period)
    gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs,
        shuffle=False, predict_class_rating=True, class_rating_split_period=split_period)

    # create_cate_model_args = {'cate_only':True, 'is_softmax':True, 'train_sc':False,
    #                                           'layers_filters':[64, 128, 256], 'dropout':0.2, 'stack_conv':2}
    # model = create_class_rating_model(create_cate_model_args, class_n=4)

    pretrained_model = load_model('C:/Users/chaya/Downloads/icon_model1.4_k0_t-ep-026-loss-1.735-acc-0.436-vloss-2.705-vacc-0.262.hdf5')        
    model = create_class_rating_model_from_pretrained_model(pretrained_model, 4)

    model.fit_generator(gen_train,
    steps_per_epoch=math.ceil(len(aial_train)/batch_size),
    validation_data=gen_test,
    validation_steps=math.ceil(len(aial_test)/batch_size),
    epochs=epochs)