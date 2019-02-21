from overall_train import train
import numpy as np
from overall_util import _prepare_limit_class_dataset, cross_validation_generator, _convert_features_and_labels_to_np_array

def train_cross_validation(k, is_regression, dense_level, epochs, batch_size, dropout_rate, limit_class, fixed_random_seed, optimizer):
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
         dropout_rate=dropout_rate, splited_dataset=splited_dataset, optimizer=optimizer, shut_up=True)
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

def test_10_times(is_regression):
    accs=[]
    for i in range(10):
        dat_train, dat_test = _prepare_limit_class_dataset(is_regression=is_regression,fixed_random_seed=False, limit_class={0:1500,1:1500,2:1500,3:1500}, use_odd=False)
        print(len(dat_train),len(dat_test))
        dat_train = _convert_features_and_labels_to_np_array(dat_train)
        dat_test = _convert_features_and_labels_to_np_array(dat_test)
        splited_data = dat_train[0], dat_train[1], dat_train[2], dat_train[3], dat_train[4], \
            dat_test[0], dat_test[1], dat_test[2], dat_test[3], dat_test[4]
        acc=train(is_regression=is_regression, dense_level=5, dropout_rate=0.05, epochs=30, batch_size=64, optimizer='adadelta',
            splited_dataset=splited_data, shut_up=True)
        print(' acc = %.3f' %  (acc,))
        accs.append(acc)
    accs = np.asarray(accs)
    if is_regression:
        best_acc = np.min(accs)
    else:
        best_acc = np.max(accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    print('%.3f %.3f %.3f' % (best_acc, avg_acc, std_acc))
    return best_acc, avg_acc, std_acc

test_10_times(True)
    


# for i in ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']:
#     result = train_cross_validation(k=5, is_regression=True, dense_level=5, batch_size=64, epochs=30,
#         dropout_rate=0.05 , limit_class={0:1500,1:1500,2:1500,3:1500}, fixed_random_seed=True, optimizer=i)
#     print('param :', i)
#     print('%.3f %.3f %.3f' % result)
#[8,16,32,64]
#[0.05, 0.1, 0.15, 0.2]
# ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam']
