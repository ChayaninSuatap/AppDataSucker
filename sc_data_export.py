from keras.models import load_model
import icon_util
import numpy as np
import mypath
import collections

def predict_for_spreadsheet(model_path, k_iter, aial_test, sc_dict):
    f = open('sc_fold' + str(k_iter) + '_testset.txt', 'w')
    f.close()
    model = load_model(model_path)
    for app_id, _ , label in aial_test:
        if (app_id in sc_dict) == False: continue
        truth = np.array(label).argmax()
        #output
        print(app_id, truth, end=' ')
        file_out = app_id + ' ' + str(truth) + ' '

        for ss_fn in sc_dict[app_id]:
            try:
                icon = icon_util.load_icon_by_fn(mypath.screenshot_folder+ss_fn, 256, 160, rotate_for_sc=True)
            except:
                continue
            icon = icon.astype('float32')
            icon /= 255
            pred = model.predict(np.array([icon]))
            t = pred[0].argmax()
            #save predict prob
            predict_prob = pred[0][t]
            #output
            print(t, predict_prob, end=' ')
            file_out += str(t) + ' ' + str(predict_prob) + ' '
        #output
        print('')
        f = open('sc_fold' + str(k_iter) + '_testset.txt', 'a')
        f.writelines(file_out + '\n')
        f.close()

def compute_mode_from_spreadsheet_txt(k_iter):
    f = open('sc_fold%d_testset.txt' % (k_iter,), 'r')
    f_vote = open('ss_vote.txt', 'w')
    f_vote.close()
    f_vote = open('ss_vote.txt', 'a')
    for line in f:
        s = line.split(' ')
        s = s[2:-1]
        #make total prob_dict
        labels = []
        prob_dict = {}
        for i in range(0, len(s), 2):
            class_label = int(s[i])
            labels.append( class_label)
            pred_prob = float(s[i+1])
            #add in prob_dict
            if class_label not in prob_dict:
                prob_dict[class_label] = pred_prob
            else:
                prob_dict[class_label] += pred_prob
            #prob dict got total not average

        #counter
        counter = collections.Counter(labels)
        ls = []
        for elem, freq in counter.items():
            ls.append(( elem, freq))
        #make prob_dict average
        for k in prob_dict.keys():
            prob_dict[k] = prob_dict[k] / counter[k]
        #get max pred occurrence
        ls_sorted = sorted(ls, key=lambda x: x[1], reverse=True)
        #compute mode
        occur_prob = [0.03965584164,0.01720791823,0.09011515073,0.05776943977,0.007892353474,0.03448052788,0.08008798033,0.1359813689,0.07161340406,0.04948893777,0.06035709665,0.03752102471,0.05647561133,0.0478069608,0.03525682495,0.1183853021,0.0599042567]
        def compute_mode(ls_sorted):
            if len(ls_sorted) == 0:
                return -1
            elif len(ls_sorted) == 1:
                return ls_sorted[0][0]
            #case top two freq is equal
            elif ls_sorted[0][1] == ls_sorted[1][1]: 
                elem0 = ls_sorted[0][0]
                elem1 = ls_sorted[1][0]
                #decide by occurrence in dataset
                # if occur_prob[elem0] > occur_prob[elem1]:
                #     return elem0
                # else:
                #     return elem1

                #decide by pred prob
                if prob_dict[elem0] * occur_prob[elem0] > prob_dict[elem1] * occur_prob[elem1]:
                    return elem0
                else:
                    return elem1
                
            elif ls_sorted[0][1] == ls_sorted[1][1] and ls_sorted[0][1] == ls_sorted[2][1]:
                raise ValueError('THREE OCCURRENCE EQUAL')
                input()
            else:
                return ls_sorted[0][0]
        
        mode = compute_mode(ls_sorted)
        f_vote.writelines(str(mode)+'\n')
        print(mode)
    f_vote.close()

if __name__ == '__main__':
    compute_mode_from_spreadsheet_txt(0)