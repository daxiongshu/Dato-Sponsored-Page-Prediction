import pandas as pd
import numpy as np
from scipy import sparse
import sys
import cPickle as pickle
name=sys.argv[1]
if name!='test.csv':
    num=name[5]
    train=pd.read_csv(name)
    y=np.array(train['sponsored'])
    ID=np.array(train['file'])
    train.drop(['sponsored','file'],inplace=True,axis=1)
    X=sparse.csr_matrix(train.values)
    del train
    print name,X.shape,y.shape
    pickle.dump(y,open('y_%s'%num,'w'))
    pickle.dump(ID,open('trainid_%s'%num,'w'))
    pickle.dump(X,open('Xtrain_%s'%num,'w'))
else:
    test=pd.read_csv(name,index_col='file')
    Xt=sparse.csr_matrix(test.values)
    ID=np.array(test.index)
    pickle.dump(ID,open('testid','w'))
    pickle.dump(Xt,open('Xtest','w'))
    print name,Xt.shape
