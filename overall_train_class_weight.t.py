from overall_train import train, train_cross_validation
from overall_util import prepare_dataset, _prepare_limit_class_dataset, cross_validation_generator

dat,_ = _prepare_limit_class_dataset(fixed_random_seed=False, limit_class={}, use_odd=False, is_regression=False)
#calc class weight
class_freq={0:0, 1:0, 2:0 ,3:0}
for x in dat:
    class_freq[x[-1]] += 1
sum_class_freq = sum(v for k,v in class_freq.items())
class_weight={}
for k,v in class_freq.items():
    class_weight[k] = v/sum_class_freq

#train
print(train_cross_validation(k=5, is_regression=False, fixed_random_seed=False, limit_class={},
 dense_level=5, epochs=30, batch_size=64, dropout_rate=0.15, optimizer='adam',
     class_weight=class_weight))
