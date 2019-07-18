from tensorflow.keras.models import load_model
import numpy as np
import mypath
import icon_util
import global_util

def predict_for_spreadsheet(model, k_iter, aial_test):
    output_path = 'icon_fold' + str(k_iter) + '_testset.txt'
    f = open(output_path, 'w')
    f.close()
    for app_id, _ , label in aial_test:
        #output
        print(app_id, end=' ')
        file_out = app_id + ' '

        try:
            icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
            icon = icon.astype('float32')
            icon /= 255
            pred = model.predict(np.array([icon]))
            t = pred[0].argmax()
            #output
            print(t, end=' ')
            file_out += str(t)
        except:
            print(-1, end=' ')
            file_out += str(-1)
        #output
        print('')
        f = open(output_path, 'a')
        f.writelines(file_out + '\n')
        f.close()

#predict icon + screenshot
def predict_combine_v1(icon_model, k_iter, aial_test, fn_postfix=''):
    #read sum_pred.txt
    sc_sum_pred_fn = 'sc_fold_sum_pred_' + str(k_iter) + '_testset_' + fn_postfix + '.txt'
    pred_combine_fn = 'pred_combine_' + str(k_iter) + '_' + fn_postfix + '.txt'
    f_sc_sum_pred = open(sc_sum_pred_fn, 'r')    
    f_pred_combine = open(pred_combine_fn, 'w')
    f_pred_combine.close()
    app_ids = []
    sc_preds = []
    for line in f_sc_sum_pred:
        splited = line.split(' ')[:-1]
        app_ids.append(splited[0])
        sc_pred = np.array([float(x) for x in splited[1:]])
        #normalize sc_pred
        sc_pred = sc_pred / sc_pred.sum()
        sc_preds.append(sc_pred)
    #predict prop of icon for each icon
    for i,app_id in enumerate(app_ids):
        try:
            icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
        except:
            print(-1)
            f_pred_combine = open(pred_combine_fn, 'a')
            f_pred_combine.write(app_id + ' -1' + ' \n')
            f_pred_combine.close()
            continue
            
        icon = icon.astype(np.float32)
        icon/=255
        pred = icon_model.predict(np.array([icon]))

        try:
            total_pred = pred + sc_preds[i]
        except:
            print(-1)
            f_pred_combine = open(pred_combine_fn, 'a')
            f_pred_combine.write(app_id + ' -1' + ' \n')
            f_pred_combine.close()
            continue

        print(total_pred.argmax())
        f_pred_combine = open(pred_combine_fn, 'a')
        f_pred_combine.writelines(app_id + ' ' + str(total_pred.argmax()) + ' \n')
        f_pred_combine.close()
    
def vote_by_maturity_for_human_test(models):
    pred_argmaxs = []
    preds = []
    for model in models:
        #eval for human test
        o = global_util.load_pickle('app_ids_for_human_test.obj')
        xs = []
        ys = []
        for app_id, class_num in o:
            icon = icon_util.load_icon_by_app_id(app_id, 128, 128)
            icon = icon.astype('float32')
            icon/=255
            xs.append(icon)
            y = [0] * 17
            y[class_num] = 1
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        pred = model.predict(xs)
        pred_argmax = pred.argmax(axis=1)
        print(len(pred))
        print(pred_argmax.shape)
        # input()
        #add pred
        pred_argmaxs.append( pred_argmax)
        preds.append( pred)
    output = []
    for i in range(len(pred_argmaxs[0])):
        if pred_argmaxs[0][i] == pred_argmaxs[1][i]:
            output.append(pred_argmaxs[0][i])
        elif pred_argmaxs[1][i] == pred_argmaxs[2][i]:
            output.append(pred_argmaxs[1][i])
        elif pred_argmaxs[0][i] == pred_argmaxs[2][i]:
            output.append(pred_argmaxs[2][i])
        else:
            ix0 = pred_argmaxs[0][i]
            ix1 = pred_argmaxs[1][i]
            ix2 = pred_argmaxs[2][i]
            if preds[0][i][ix0] >= preds[1][i][ix1] and preds[0][i][ix0] >= preds[2][i][ix2]:
                output.append( pred_argmaxs[0][i])
            elif preds[1][i][ix1] >= preds[0][i][ix0] and preds[1][i][ix1] >= preds[2][i][ix2]:
                output.append( pred_argmaxs[1][i])
            elif preds[2][i][ix2] >= preds[1][i][ix1] and preds[2][i][ix2] >= preds[0][i][ix0]:
                output.append( pred_argmaxs[2][i])
    [print(x) for x in output]


if __name__ == '__main__' :
    model_fns = [
        'cate_conv_512_k0-ep-379-loss-0.022-acc-0.994-vloss-4.943-vacc-0.353.hdf5',
        'cate_conv_1024_k0-ep-609-loss-0.015-acc-0.995-vloss-5.739-vacc-0.361.hdf5',
        'cate_model5_cw-ep-732-loss-0.016-acc-0.995-vloss-5.906-vacc-0.377.hdf5'
    ]
    models = [load_model(model_fn) for model_fn in model_fns]
    vote_by_maturity_for_human_test(models)
