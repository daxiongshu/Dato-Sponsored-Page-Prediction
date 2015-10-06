import pickle
import sys
from sklearn import metrics
import pandas as pd
import numpy as np

def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
name=sys.argv[1]
name2=sys.argv[2]
s1=pd.read_csv(name)
s2=pd.read_csv(name2)
y=pickle.load(open('y_va.p'))
yp=np.array(s1['sponsored']).argsort().argsort()*1.0/s1.shape[0]
yp2=np.array(s2['sponsored']).argsort().argsort()*1.0/s1.shape[0]
print myauc(y,yp)

for i in range(21):
    #yy=yp*(0.9+i*0.01)+yp2*(0.1-i*0.01)
    yy=yp*i+yp2*(100-i)
    yy/=100
    print i,myauc(y,yy)
