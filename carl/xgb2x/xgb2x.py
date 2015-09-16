from xgb_classifier import xgb_classifier
from sklearn.cross_validation import KFold,StratifiedKFold
from sklearn import preprocessing
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from sklearn import metrics
def train_predict(X,y,Xt,yt=[],c=1):
    if c=='xgb':
        clf=xgb_classifier(num_round=200,eta=0.1,min_child_weight=2,depth=20, subsample=1,col=0.6)
        return clf.train_predict(X,y,Xt,yt)
    if c=='rf':
        clf=RandomForestClassifier(n_estimators=200,n_jobs=-1,max_depth=13,min_samples_split=4,min_samples_leaf=9, max_leaf_nodes= 1100)
        clf.fit(X,y)
        return clf.predict_proba(Xt).T[1]    
    if c=='rf1':
        clf=RandomForestClassifier(n_estimators=1000,n_jobs=-1)
        clf.fit(X,y)
        return clf.predict_proba(Xt).T[1]
import pickle
def myauc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
def kfold_cv(X_train, y_train,k,cl='xgb',xid=None):

    kf=StratifiedKFold(y_train,n_folds=k)
    #pickle.dump(kf,open('kf.p','w'))
    xx=[]
    pred=y_train.copy().astype(float)

    for train_index, test_index in kf:
        X_train_cv, X_test_cv = X_train[train_index,:],X_train[test_index,:]
        y_train_cv, y_test_cv = y_train[train_index],y_train[test_index]
        yp=train_predict(X_train_cv,y_train_cv,X_test_cv,yt=y_test_cv,c=cl)
        #xx.append(normalized_weighted_gini(y_test_cv,yp))
        xx.append(myauc(y_test_cv,yp))
        pred[test_index]=yp
        print xx[-1]
    print xx,' mean:',np.mean(xx)
    return pred
    print 'overall auc' ,myauc(y_train,pred)

import cPickle as pickle
from scipy import sparse
y=pickle.load(open('ytrain.p'))#np.array(train['sponsored']).astype(float)
idx=pickle.load(open('testid'))
X=sparse.csr_matrix(pickle.load(open('Xtrain.p')))
Xt=pickle.load(open('Xtest'))




print X.shape,y.shape,Xt.shape
cl='xgb'
yp=train_predict(X,y,Xt,c=cl)
s=pd.DataFrame({'file':idx,'sponsored':yp})

s.to_csv('xgb2x.csv',index=False)
