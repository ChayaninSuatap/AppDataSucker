from overall_train import train, train_cross_validation
from overall_util import prepare_dataset, _prepare_limit_class_dataset, cross_validation_generator
from keras_util import compute_class_weight

dat,_ = _prepare_limit_class_dataset(fixed_random_seed=False, limit_class={}, use_odd=False, is_regression=False)
#calc class weight
class_weight = compute_class_weight([x[-1] for x in dat])
print(class_weight)
#train
print(train_cross_validation(k=5, is_regression=False, fixed_random_seed=False, limit_class={},
 dense_level=5, epochs=30, batch_size=64, dropout_rate=0.15, optimizer='adam',
     class_weight=class_weight, save_xy=True))
