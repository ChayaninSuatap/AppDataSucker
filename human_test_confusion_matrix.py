from sklearn.metrics import confusion_matrix
gt=[]
print('input ground truth')
for i in range(340):
    x = input()
    gt.append(int(x))
print(gt)

pred=[]
print('input predict')
for i in range(340):
    x = input()
    pred.append(int(x))
print(pred)

print(confusion_matrix(gt, pred))
