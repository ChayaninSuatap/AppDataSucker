import numpy as np
import overall_db_util
from overall_feature_util import extract_feature_vec
import random
from keras.layers import Input, Dense, concatenate
from keras.models import Model
import matplotlib.pyplot as plt

def save_prediction_to_file(model, dataset, batch_size):
    answers = [x[0] for x in model.predict(dataset, batch_size=batch_size)]
    answers = []
    for x in model.predict(dataset, batch_size=batch_size):
        answers.append(np.argmax(x))

    f = open('answers.txt','w')
    for x in answers:
        f.write(str(x)+'\n')

def save_testset_labels_to_file(testset):
    f = open('testset_labels.txt','w')
    for x in testset:
        f.write(str(x)+'\n')

def plot_loss(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graph_acc.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('graph_loss.png')

def prepare_dataset(use_download_amount=True, use_rating_amount=True, testset_percent=90):
    #query from db
    features_and_labels = []
    for record in overall_db_util.query():
        t = extract_feature_vec(record, use_download_amount=use_download_amount, use_rating_amount=use_rating_amount)
        features_and_labels.append(t)
    #shuffle
    random.seed(1)
    random.shuffle(features_and_labels)
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
    #split train, test
    split_num = int(len(features_and_labels) * 90/100)
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

def create_model(input_other_shape, input_category_shape, input_sdk_version_shape, input_content_rating_shape, \
    dense_level, num_class, category_densed_shape=10, sdk_version_densed_shape=10, content_rating_densed_shape=10):
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
    for i in range(dense_level):
        t = Dense(dense_size, activation='relu', name='overall_dense_'+str(i))(t)
    #output layer
    output_layer = Dense(num_class, activation='softmax', name='overall_output')(t)
    #create model
    model = Model(inputs=[category_input, sdk_version_input, content_rating_input, other_input], outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model