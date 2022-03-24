from tokenize import Double
import numpy as np

#getScore takes to numpy arrays of equall size one with the predictions and one with the labels and computes the score#
#Only 0 and 1 inputs are accepted
def getScore(pred, res):
    FP = 0; TP = 0; FN = 0; TN = 0
    for i in range(len(pred)):
        if pred[i] == 1 and res[i] == 0:
            FP += 1
        elif pred[i] == 1 and res[i] == 0:
            FN += 1
        elif pred[i] == 1 and res[i] == 1: 
            TP += 1
        else: 
            TN += 1
    precision = TP /float(TP+FP)
    recall = TP / float(TP+FN)
    f1Score = (2*precision*recall)/(precision+recall)
    return f1Score



#a = np.array([1,0,1,0,1,1,1])
#b = np.array([0,0,0,0,1,1,1])
#s = getScore(a,b)
#print(s)