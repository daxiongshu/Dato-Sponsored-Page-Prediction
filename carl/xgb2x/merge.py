from scipy import sparse
import cPickle as pickle
import numpy as np
xx=[]
yy=[]
idx=[]
for i in range(5):
    xx.append(pickle.load(open('Xtrain_%d'%i)))
    yy.append(pickle.load(open('y_%d'%i)))
    idx.append(pickle.load(open('trainid_%d'%i)))
xx=sparse.vstack(xx)
yy=np.concatenate(yy)
idx=np.concatenate(idx)
pickle.dump(xx,open('Xtrain.p','w'))
pickle.dump(yy,open('ytrain.p','w'))
pickle.dump(idx,open('trainid.p','w'))

