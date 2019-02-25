from overall_train import train
import numpy as np
from overall_util import _prepare_limit_class_dataset, cross_validation_generator, _convert_features_and_labels_to_np_array



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
