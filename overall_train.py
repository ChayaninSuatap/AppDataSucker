import overall_db_util
from overall_feature_util import extract_feature_vec
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from overall_util import save_prediction_to_file, save_testset_labels_to_file, prepare_dataset, create_model, print_dataset_freq, cross_validation_generator, _prepare_limit_class_dataset
from keras.callbacks import ModelCheckpoint
from plt_util import plot_loss
from keras import backend as K; K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

def train(is_regression, dense_level, epochs, batch_size, dropout_rate, splited_dataset, optimizer, shut_up=False, class_weight=None):

    x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90, y_90, \
        x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10, y_10 = splited_dataset
    
    if not is_regression:
        #print frequency of each class
        if not shut_up:
            print_dataset_freq(y_90, 'y_90 freq')
            print_dataset_freq(y_10, 'y_10 freq')
        y_10 = to_categorical(y_10,4)
        y_90 = to_categorical(y_90,4)

    #get x features shape
    other_shape = x_other_10[0].shape[0]
    category_shape = x_category_10[0].shape[0]
    sdk_version_shape = x_sdk_version_10[0].shape[0]
    content_rating_shape = x_content_rating_10[0].shape[0]
    if not shut_up:
        print(other_shape, category_shape, sdk_version_shape, content_rating_shape)

    model = create_model(input_other_shape=other_shape, input_category_shape=category_shape, input_sdk_version_shape=sdk_version_shape, \
    input_content_rating_shape=content_rating_shape, dropout_rate=dropout_rate,
    category_densed_shape=10, sdk_version_densed_shape=10, content_rating_densed_shape=10, optimizer=optimizer,
     dense_level=dense_level, num_class=4,is_regression=is_regression)

    #assign checkpoint
    if not is_regression:
        checkpoint_path = 'models/overall_ep_{epoch:03d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_val_acc_{val_acc:.2f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=False)
    else:
        checkpoint_path = 'models/overall_ep_{epoch:03d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_val_mae_{val_mean_absolute_error:.2f}.hdf5'
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mean_absolute_error', save_best_only=False)
    #train
    history = model.fit([x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90], y_90, verbose=(not shut_up),
    validation_data=([x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10], y_10), epochs=epochs, batch_size=batch_size,
     callbacks=[checkpoint], class_weight=class_weight)
    #return result
    if is_regression:
        max_acc = min(history.history['val_mean_absolute_error']) #mae
    else:
        max_acc = max(history.history['val_acc'])
    if not shut_up : print('best acc : %.3f' % (max_acc,))
    plot_loss(history, is_regression)
    return max_acc

def train_cross_validation(k, is_regression, dense_level, epochs, batch_size, dropout_rate, limit_class, fixed_random_seed, optimizer, class_weight=None):
    #prepare limit class data
    prepared_limit_dat, _ = _prepare_limit_class_dataset(is_regression=is_regression,
    fixed_random_seed=fixed_random_seed, limit_class=limit_class, use_odd=False)
    #prepare k chrunks generator
    chrunks_gen = cross_validation_generator(k, prepared_limit_dat)
    #train each chrunks
    accs = []
    for i, splited_dataset in enumerate(chrunks_gen):
        print('training pass ',i, end='')
        max_acc = train(is_regression=is_regression, dense_level=dense_level, epochs=epochs, batch_size=batch_size,
         dropout_rate=dropout_rate, splited_dataset=splited_dataset, optimizer=optimizer, shut_up=True, class_weight=class_weight)
        accs.append(max_acc)
        print(' acc = %.3f' %  (max_acc,))
    #finalize result
    accs = np.asarray(accs)
    if is_regression:
        best_acc = np.min(accs)
    else:
        best_acc = np.max(accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    return best_acc, avg_acc, std_acc

if __name__ == '__main__':
    splited_dataset = prepare_dataset(is_regression=False,
        fixed_random_seed=False, testset_percent=90,
        limit_class={0:1500,1:1500,2:1500,3:1500}, use_odd=False)
    print(train(is_regression=False, dense_level=5, dropout_rate=0, epochs=30, batch_size=32, optimizer='adam',
     splited_dataset=splited_dataset))
