import random
import numpy as np
import preprocess_util
import keras_util
import math
import icon_cate_util

def make_icon_dataset_generator(k_iter, batch_size, epochs, enable_cache=False, yield_app_id=False):
    aial_train, aial_test = prepare_aial_train_test(k_iter)

    gen_train = icon_cate_util.datagenerator(aial_train, batch_size, epochs, cate_only=True, enable_cache=enable_cache, yield_app_id=yield_app_id)
    gen_test = icon_cate_util.datagenerator(aial_test, batch_size, epochs, cate_only=True, shuffle=False, enable_cache=enable_cache, yield_app_id=yield_app_id)

    train_steps = math.ceil( len(aial_train) / batch_size)
    test_steps = math.ceil( len(aial_test) / batch_size)

    return gen_train, gen_test, train_steps, test_steps

def prepare_aial_train_test(k_iter):
    random.seed(859)
    np.random.seed(859)
    aial = preprocess_util.prep_rating_category_scamount_download(for_softmax=True)
    aial = preprocess_util.remove_low_rating_amount(aial, 100)
    random.shuffle(aial)
    aial = preprocess_util.get_app_id_rating_cate_from_aial(aial)

    aial_train, aial_test = keras_util.gen_k_fold_pass(aial, kf_pass= k_iter, n_splits=4)
    return aial_train, aial_test

if __name__ == '__main__':
    aial_train, aial_test = prepare_aial_train_test(0)
    print(aial_train)