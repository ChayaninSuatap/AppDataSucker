import overall_db_util
from overall_feature_util import extract_feature_vec
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from overall_util import save_prediction_to_file, save_testset_labels_to_file, prepare_dataset, create_model

x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90, y_90, \
    x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10, y_10 = prepare_dataset()

y_10 = to_categorical(y_10,4)
y_90 = to_categorical(y_90,4)
other_shape = x_other_10[0].shape[0]
category_shape = x_category_10[0].shape[0]
sdk_version_shape = x_sdk_version_10[0].shape[0]
content_rating_shape = x_content_rating_10[0].shape[0]
print(other_shape, category_shape, sdk_version_shape, content_rating_shape)

model = create_model(input_other_shape=other_shape, input_category_shape=category_shape, input_sdk_version_shape=sdk_version_shape, \
    input_content_rating_shape=content_rating_shape, \
    category_densed_shape=18, sdk_version_densed_shape=38, content_rating_densed_shape=38, dense_level=10, num_class=4)
model.fit([x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90], y_90, \
    validation_data=([x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10], y_10), epochs=100, batch_size=32)