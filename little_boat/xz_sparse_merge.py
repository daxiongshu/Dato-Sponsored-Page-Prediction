import pandas as pd
import numpy as np
from scipy import sparse
import pickle
xx = []
ID = []
for i in xrange(16):
    print "doing %i file..."%i
    path = 'xz_tokens_%i.csv'%i
    final = pd.read_csv(path)
    ID.append(final['file'].values)
    del final['file']
    fea=sparse.csr_matrix(final.values)
    xx.append(fea)

print "merge them..."
xx=sparse.vstack(xx)
ID=np.concatenate(ID)
print xx.shape
print ID.shape
pickle.dump(xx,open('xz_tokens_01.p','w'))
pickle.dump(ID,open('xz_tokens_id.p','w'))
