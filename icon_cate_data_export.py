from tensorflow.keras.models import load_model
import numpy as np
import mypath
import icon_util

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