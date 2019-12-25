import icon_cate_util
import icon_util
import numpy as np
from keras.utils import to_categorical
import global_util

#create cw models
# model_i2 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512])
# model_i3 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1028])
# model_i5 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512], stack_conv=2)
# model_i7 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1024], dropout=0)
# model_i9 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1024], sliding_dropout=(0.05, 0.05))

#create not cw models
# model_i2 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512])
# model_i3 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1024])
# model_i5 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512], stack_conv=2)
# model_i7 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1024], dropout=0)
# model_i9 = icon_cate_util.create_icon_cate_model(cate_only=True, is_softmax=True, train_sc=False, layers_filters = [64, 128, 256, 512, 1024], sliding_dropout=(0.05, 0.05))


#load dataset
# xs = []
# for i in range(340):
#     icon = icon_util.load_icon_by_fn('icons_human_test/' + str(i) + '.png', 128, 128)
#     xs.append(icon)
# xs = np.array(xs) / 255

ys = []
f = open('ground_truth.txt','r')
for line in f:
    ys.append(int(line[0:-1]))
# ys = to_categorical(np.array(ys), 17)

# model_i2.load_weights('icon_ensemble_cw/cate_model_i2_cw_k0-ep-101-loss-0.077-acc-0.975-vloss-0.399-vacc-0.375.hdf5')
# model_i3.load_weights('icon_ensemble_cw/cate_model_i3_cw_k0-ep-549-loss-0.009-acc-0.998-vloss-5.685-vacc-0.357.hdf5')
# model_i5.load_weights('icon_ensemble_cw/cate_model_i5_cw_k0-ep-1537-loss-0.012-acc-0.996-vloss-5.044-vacc-0.344.hdf5')
# model_i7.load_weights('icon_ensemble_cw/cate_model_i7_cw_k0-ep-1037-loss-0.020-acc-0.998-vloss-6.140-vacc-0.329.hdf5')
# model_i9.load_weights('icon_ensemble_cw/cate_model5_fix_cw_k0-ep-192-loss-0.030-acc-0.991-vloss-5.340-vacc-0.365.hdf5')

#load not cw models
# model_i2.load_weights('icon_ensemble_cw/i2.hdf5')
# model_i3.load_weights('icon_ensemble_cw/i3.hdf5')
# model_i5.load_weights('icon_ensemble_cw/i5.hdf5')
# model_i7.load_weights('icon_ensemble_cw/i7.hdf5')
# model_i9.load_weights('icon_ensemble_cw/i9.hdf5')

# models = [model_i2, model_i3, model_i5, model_i7, model_i9]

# preds = None
# for model in models:
#     pred = model.predict(xs)
#     if preds is None:
#         preds = pred
#     else:
#         preds += pred

# print(preds)
# preds /= 5
# print(preds)
# global_util.save_pickle(preds, 'icon_ensemble_cw/preds_not_cw.obj')

preds = global_util.load_pickle('icon_ensemble_cw/preds_not_cw.obj')

total_correct = 0

for i, pred in enumerate(preds):
    if pred.argmax() == ys[i]:
        total_correct += 1
    print(pred.argmax())

print(total_correct/340)