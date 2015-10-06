import os
import glob
import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import xgboost as xgb


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)

    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                      shape=loader['shape'])


def feature_reduction(X_train, y_train, path, num_round=500, run=5):
    params = {'max_depth':25, 'eta':0.1, 'objective':'binary:logistic', 
            'eval_metric':'auc', 'silent':1}
    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    md = itertools.cycle([5*i + 15 for i in range(5)])

    for i in range(run):
        params['max_depth'] = next(md)
        params_list = params.items()
        watch_list = [(dtrain, 'eval')]
        bst = xgb.train(params_list, dtrain, num_round, watch_list)
        with open(os.path.join(path, "fscore{}.txt".format(i)), "w") as fhandle:
            fhandle.write("{}\n".format(bst.get_fscore()))


if __name__ == "__main__":

    # read training data
    train_df = pd.read_csv("data/train_v2.csv")
    y_train = train_df.sponsored.values

    for f in glob.glob("matrix/*"):
        if os.path.isdir(f):
            X_train = load_sparse_csr(os.path.join(f, "X_train.np"))
            if X_train.shape[1] > 1e3:
                print "Processing feature reduction on: {}".format(os.path.join(f, "X_train.np"))
                print "Feature number: {}".format(X_train.shape[1])
                feature_reduction(X_train, y_train, f)

