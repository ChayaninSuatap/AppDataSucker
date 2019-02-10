import numpy as np
def save_prediction_to_file(model, dataset, batch_size):
    answers = [x[0] for x in model.predict(dataset, batch_size=batch_size)]
    answers = []
    for x in model.predict(dataset, batch_size=batch_size):
        answers.append(np.argmax(x))

    f = open('answers.txt','w')
    for x in answers:
        f.write(str(x)+'\n')

def save_testset_labels_to_file(testset):
    f = open('testset_labels.txt','w')
    for x in testset:
        f.write(str(x)+'\n')
