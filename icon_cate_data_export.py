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
