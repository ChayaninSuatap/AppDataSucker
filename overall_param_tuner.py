from overall_train import train
import numpy as np
from overall_util import _prepare_limit_class_dataset, cross_validation_generator

def train_cross_validation(k, is_regression, dense_level, epochs, batch_size, dropout_rate, limit_class, fixed_random_seed):
    #prepare limit class data
    prepared_limit_dat = _prepare_limit_class_dataset(is_regression=is_regression, fixed_random_seed=fixed_random_seed, limit_class=limit_class, use_odd=False)
    #prepare k chrunks generator
    chrunks_gen = cross_validation_generator(k, prepared_limit_dat)
    #train each chrunks
    accs = []
    for i, splited_dataset in enumerate(chrunks_gen):
        print('training pass ',i, end='')
        max_acc = train(is_regression=is_regression, dense_level=dense_level, epochs=epochs, batch_size=batch_size,
         dropout_rate=dropout_rate, splited_dataset=splited_dataset, shut_up=True)
        accs.append(max_acc)
        print(' acc = %.3f' %  (max_acc,))
    accs = np.asarray(accs)
    best_acc = np.max(accs)
    avg_acc = np.mean(accs)
    std_acc = np.std(accs)
    return best_acc, avg_acc, std_acc

result = train_cross_validation(k=5, is_regression=False, dense_level=5, batch_size=32, epochs=15,
    dropout_rate=0, limit_class={0:1500,1:1500,2:1500,3:1500}, fixed_random_seed=False)
print('%.3f %.3f %.3f' % result)
