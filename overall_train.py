import overall_db_util
from overall_feature_util import extract_feature_vec
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from overall_util import save_prediction_to_file, save_testset_labels_to_file, prepare_dataset, create_model, print_dataset_freq
from keras.callbacks import ModelCheckpoint
from plt_util import plot_loss
from keras import backend as K; K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

is_regression=False

x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90, y_90, \
    x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10, y_10 = prepare_dataset(is_regression=is_regression, fixed_random_seed=False,
    limit_class={0:1500,1:1500,2:1500,3:1500},use_download_amount=True,use_rating_amount=True)

if not is_regression:
    #print frequency of each class
    print_dataset_freq(y_90, 'y_90 freq')
    y_10 = to_categorical(y_10,4)
    y_90 = to_categorical(y_90,4)

#get x features shape
other_shape = x_other_10[0].shape[0]
category_shape = x_category_10[0].shape[0]
sdk_version_shape = x_sdk_version_10[0].shape[0]
content_rating_shape = x_content_rating_10[0].shape[0]
print(other_shape, category_shape, sdk_version_shape, content_rating_shape)

model = create_model(input_other_shape=other_shape, input_category_shape=category_shape, input_sdk_version_shape=sdk_version_shape, \
    input_content_rating_shape=content_rating_shape, \
    category_densed_shape=10, sdk_version_densed_shape=10, content_rating_densed_shape=10, dense_level=10, num_class=4,is_regression=is_regression)

#assign checkpoint
if not is_regression:
    checkpoint_path = 'models/overall_ep_{epoch:03d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_val_acc_{val_acc:.2f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', save_best_only=False)
else:
    checkpoint_path = 'models/overall_ep_{epoch:03d}_loss_{loss:.3f}_val_loss_{val_loss:.3f}_val_mae_{val_mean_absolute_error:.2f}.hdf5'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_mean_absolute_error', save_best_only=False)

history = model.fit([x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90], y_90, \
    validation_data=([x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10], y_10), epochs=30, batch_size=32, callbacks=[checkpoint])
print('best acc :', max(history.history['val_acc']))
plot_loss(history, is_regression)
