import numpy as np
import math
import overall_db_util
from overall_feature_util import extract_feature_vec, normalize_number
import random
from keras.layers import Input, Dense, concatenate, Activation, Dropout
from keras.models import Model
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from datetime import datetime

def save_prediction_to_file(model, dataset, is_regression, batch_size):
    answers = []
    for x in model.predict(dataset, batch_size=batch_size):
        if not is_regression:
            answers.append(np.argmax(x))
        else:
            answers.append(x[0])

    f = open('answers.txt','w')
    for x in answers:
        f.write(str(x)+'\n')

def save_testset_labels_to_file(testset):
    f = open('testset_labels.txt','w')
    for x in testset:
        f.write(str(x)+'\n')

def _prepare_limit_class_dataset(fixed_random_seed, limit_class, use_odd, is_regression):
    #seed
    if fixed_random_seed: random.seed(7)
    else: random.seed(datetime.now())
    #init current class num
    features_and_labels = []
    current_class_num = {}
    for key in limit_class:
        current_class_num[key] = 0
    #query from db
    queried = list(overall_db_util.query())
    #shuffle for limit class number
    random.shuffle(queried)
    for record in queried:
        t = extract_feature_vec(record, use_download_amount=use_odd, use_rating_amount=use_odd, is_regression=is_regression)
        class_label = t[-1]
        if class_label in limit_class:
            if current_class_num[class_label] <= limit_class[class_label]:
                current_class_num[class_label] += 1
                features_and_labels.append(t)
        else:
            features_and_labels.append(t)
    #shuffle for vary test and train set
    random.shuffle(features_and_labels)
    return features_and_labels

def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cross_validation_generator(k, prepared_limit_dataset):
    "use after prepare_limit_class_dataset"
    chrunk_size = math.ceil(len(prepared_limit_dataset)/k)
    #split features and labels
    feat_category = []
    feat_sdk_version = []
    feat_content_rating = []
    feat_other = []
    labels = []
    for onehot, single_node, label in prepared_limit_dataset:
        feat_category.append(onehot[0])
        feat_sdk_version.append(onehot[1])
        feat_content_rating.append(onehot[2])
        feat_other.append(single_node)
        labels.append(label)
    #chrunk feats and labels
    feat_category_chrunk = list(_chunks(feat_category, chrunk_size))
    feat_sdk_version_chrunk = list(_chunks(feat_sdk_version, chrunk_size))
    feat_content_rating_chrunk = list(_chunks(feat_content_rating, chrunk_size))
    feat_other_chrunk = list(_chunks(feat_other, chrunk_size))
    labels_chrunk = list(_chunks(labels, chrunk_size))
    #generate
    for i in range(k):
        #prepare train set
        cate_90 ,sdk_90, content_90, other_90, label_90 = [], [], [], [], []
        cate_10, sdk_10, content_10, other_10, label_10 = [], [], [], [], []
        for j in range(k):
            #find chrunk n that is not "test chrunk"
            if j != i:
                cate_90 += feat_category_chrunk[j]
                sdk_90 += feat_sdk_version_chrunk[j]
                content_90 += feat_content_rating_chrunk[j]
                other_90 += feat_other_chrunk[j]
                label_90 += labels_chrunk[j]
            else:
                #test chrunk
                cate_10 += feat_category_chrunk[j]
                sdk_10 += feat_sdk_version_chrunk[j]
                content_10 += feat_content_rating_chrunk[j]
                other_10 += feat_other_chrunk[j]
                label_10 += labels_chrunk[j]
        #generate a pass of cross validation
        cate_90 = np.asarray(cate_90)
        sdk_90 = np.asarray(sdk_90)
        content_90 = np.asarray(content_90)
        other_90 = np.asarray(other_90)
        label_90 = np.asarray(label_90)
        cate_10 = np.asarray(cate_10)
        sdk_10 = np.asarray(sdk_10)
        content_10 = np.asarray(content_10)
        other_10 = np.asarray(other_10)
        label_10 = np.asarray(label_10)
        yield cate_90, sdk_90, content_90, other_90, label_90, cate_10, sdk_10, content_10, other_10, label_10
 
def prepare_dataset(is_regression, fixed_random_seed, limit_class, use_odd, testset_percent):
    #limit class = dict of key = class number , value = limit number
    #ex. {2:5000} : limit class 2 by 5000
    features_and_labels = _prepare_limit_class_dataset(fixed_random_seed=fixed_random_seed, limit_class=limit_class,
        use_odd=use_odd, is_regression=is_regression)
    #split features and labels
    feat_category = []
    feat_sdk_version = []
    feat_content_rating = []
    feat_other = []
    labels = []
    for onehot, single_node, label in features_and_labels:
        feat_category.append(onehot[0])
        feat_sdk_version.append(onehot[1])
        feat_content_rating.append(onehot[2])
        feat_other.append(single_node)
        labels.append(label)
    #normalize
    feat_other = normalize_number(feat_other)
    #split train, test
    split_num = int(len(features_and_labels) * testset_percent/100)
    x_category_90 = np.asarray(feat_category[:split_num])
    x_sdk_version_90 = np.asarray(feat_sdk_version[:split_num])
    x_content_rating_90 = np.asarray(feat_content_rating[:split_num])
    x_other_90 = np.asarray(feat_other[:split_num])
    y_90 = np.asarray(labels[:split_num])

    x_category_10 = np.asarray(feat_category[split_num:])
    x_sdk_version_10 = np.asarray(feat_sdk_version[split_num:])
    x_content_rating_10 = np.asarray(feat_content_rating[split_num:])
    x_other_10 = np.asarray(feat_other[split_num:])
    y_10 = np.asarray(labels[split_num:])

    return x_category_90, x_sdk_version_90, x_content_rating_90, x_other_90, y_90 ,\
        x_category_10, x_sdk_version_10, x_content_rating_10, x_other_10, y_10

def print_dataset_freq(dataset, preprint_text=''):
    if preprint_text != '': print(preprint_text)
    for label in np.unique(dataset):
        print('class ', label, ' :', np.count_nonzero(dataset==label))

def create_model(input_other_shape, input_category_shape, input_sdk_version_shape, input_content_rating_shape, \
    
    dense_level, num_class, is_regression, dropout_rate,
     category_densed_shape=10, sdk_version_densed_shape=10, content_rating_densed_shape=10):
    #input layer
    category_input = Input(shape=(input_category_shape,), name='input_category')
    sdk_version_input = Input(shape=(input_sdk_version_shape,), name='input_sdk_version')
    content_rating_input = Input(shape=(input_content_rating_shape,), name='input_content_rating')
    other_input = Input(shape=(input_other_shape,), name='input_other')
    #minimize inputs
    category_densed = Dense(category_densed_shape, activation='relu', name='category_densed')(category_input)
    sdk_version_densed = Dense(sdk_version_densed_shape, activation='relu', name='sdk_version_densed')(sdk_version_input)
    content_rating_densed = Dense(content_rating_densed_shape, activation='relu', name='content_rating_densed')(content_rating_input)
    #concatenate inputs
    t = concatenate([category_densed, sdk_version_densed, content_rating_densed, other_input], name='overall_input_concatenated')
    #dense layers
    dense_size = category_densed_shape + sdk_version_densed_shape + content_rating_densed_shape + input_other_shape
    # print('dense size', dense_size)
    for i in range(dense_level):
        t = Dense(dense_size, activation='relu', name='overall_dense_'+str(i))(t)
        t = Dropout(dropout_rate, name='overall_dense_dropout_'+str(i))( t)
    #output layer
    #regression output layer
    if is_regression:
        #custom activation
        def my_sigmoid(x):
            return (K.sigmoid(x) * 5)
        act = Activation(my_sigmoid)
        act.__name__ = 'my_sigmoid'
        get_custom_objects().update({'my_sigmoid': act})
        output_layer = Dense(1, activation='my_sigmoid', name='overall_output')(t)
    #category output layer
    else:
        output_layer = Dense(num_class, activation='softmax', name='overall_output')(t)
    #create model
    model = Model(inputs=[category_input, sdk_version_input, content_rating_input, other_input], outputs=output_layer)
    if is_regression:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model