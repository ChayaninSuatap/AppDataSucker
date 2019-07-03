from keras.models import load_model
import icon_util
import numpy as np
import mypath
import collections

def predict_for_spreadsheet(model, k_iter, aial_test, sc_dict, fn_postfix=''):
    f = open('sc_fold' + str(k_iter) + '_testset_' + fn_postfix + '.txt', 'w')
    f_mode_5 = open('sc_fold_mode_5_' + str(k_iter) + '_testset_' + fn_postfix + '.txt', 'w')
    f.close()
    f_mode_5.close()
    for app_id, _ , label in aial_test:
        if (app_id in sc_dict) == False: continue
        truth = np.array(label).argmax()
        #output
        print(app_id, truth, end=' ')
        file_out = app_id + ' ' + str(truth) + ' '

        #accumulator for mode type 5
        pred_acc = np.array([0] * 17).astype('float64')
        have_some_sc = False

        for ss_fn in sc_dict[app_id]:
            try:
                icon = icon_util.load_icon_by_fn(mypath.screenshot_folder+ss_fn, 256, 160, rotate_for_sc=True)
            except:
                continue
            icon = icon.astype('float32')
            icon /= 255
            pred = model.predict(np.array([icon]))
            t = pred[0].argmax()
            #add pred for mode type 5
            pred_acc += pred[0]

            #save predict prob
            predict_prob = pred[0][t]
            #output
            print(t, predict_prob, end=' ')
            file_out += str(t) + ' ' + str(predict_prob) + ' '
            #flag for mode type 5
            have_some_sc=True

        #write an app_id pred
        print('')
        f = open('sc_fold' + str(k_iter) + '_testset_' + fn_postfix + '.txt', 'a')
        f.writelines(file_out + '\n')
        f.close()
        #write mode 5 result
        f_mode_5 = open('sc_fold_mode_5_' + str(k_iter) + '_testset_' + fn_postfix + '.txt', 'a')
        mode_5_result = pred_acc.argmax()
        if have_some_sc:
            f_mode_5.writelines(app_id + ' ' + str(mode_5_result) + '\n')
        else:
            f_mode_5.writelines(app_id + ' \n')
        f_mode_5.close()

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
        
        # compute mode for 4 type
        try:
            mode = max(prob_dict.items(), key = lambda x : x[1])[0]
        except:
            mode = -1
        
        # #compute mode for first 3 type
        # #counter
        # counter = collections.Counter(labels)
        # ls = []
        # for elem, freq in counter.items():
        #     ls.append(( elem, freq))
        # #make prob_dict average for 2 type
        # for k in prob_dict.keys():
        #     prob_dict[k] = prob_dict[k] / counter[k]
        # #get max pred occurrence
        # ls_sorted = sorted(ls, key=lambda x: x[1], reverse=True)
        # #compute mode
        # occur_prob = [0.03965584164,0.01720791823,0.09011515073,0.05776943977,0.007892353474,0.03448052788,0.08008798033,0.1359813689,0.07161340406,0.04948893777,0.06035709665,0.03752102471,0.05647561133,0.0478069608,0.03525682495,0.1183853021,0.0599042567]
        # def compute_mode(ls_sorted):
        #     if len(ls_sorted) == 0:
        #         return -1
        #     elif len(ls_sorted) == 1:
        #         return ls_sorted[0][0]
        #     #case top two freq is equal for type 1, 2, 3
        #     elif ls_sorted[0][1] == ls_sorted[1][1]: 
        #         elem0 = ls_sorted[0][0]
        #         elem1 = ls_sorted[1][0]
        #         #decide by occurrence in dataset
        #         # if occur_prob[elem0] > occur_prob[elem1]:
        #         #     return elem0
        #         # else:
        #         #     return elem1

        #         #decide by pred prob for type 2, 3
        #         # if prob_dict[elem0]  > prob_dict[elem1] : #for type 2
        #         if prob_dict[elem0] * occur_prob[elem0] > prob_dict[elem1] * occur_prob[elem1]: #for type 3
        #             return elem0
        #         else:
        #             return elem1
                
        #     elif ls_sorted[0][1] == ls_sorted[1][1] and ls_sorted[0][1] == ls_sorted[2][1]:
        #         raise ValueError('THREE OCCURRENCE EQUAL')
        #         input()
        #     else:
        #         return ls_sorted[0][0]
        # mode = compute_mode(ls_sorted)

        #must have every case
        f_vote.writelines(str(mode)+'\n')
        print(mode)
    f_vote.close()

def predict_for_spreadsheet_remove_prob():
    f = open('sc_fold0_testset.txt')
    f_save = open('sc_fold0_testset_remove_prob.txt', 'w')
    f_save = open('sc_fold0_testset_remove_prob.txt', 'a')
    for line in f:
        s = line.split(' ')
        tail = s[2:-1]
        out = ''
        # out = s[0] + ' ' +  s[1]
        #
        if len(tail) == 0:
            pass
        else:
            for i in range(0, len(tail), 2):
                out += ' ' + tail[i]
        out += '\n'
        print(out)
        f_save.write(out)


if __name__ == '__main__':
    compute_mode_from_spreadsheet_txt(0)
    # predict_for_spreadsheet_remove_prob()