import os
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.sparse as sparse


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)

    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                      shape=loader['shape'])


def sum_max_feat(df, matrix_fn, prefix):
    with open(matrix_fn) as fhandle:
        X = load_sparse_csr(fhandle)
        df["{}_sum".format(prefix)] = X.sum(axis=1)
        df["{}_max".format(prefix)] = X.max(axis=1).todense()
        return df


if __name__ == "__main__":

    # read test and train id into data frames
    train_df = pd.read_csv("data/train_v2.csv")
    test_df = pd.read_csv("data/sampleSubmission_v2.csv")

    # matrix filenames to be process
    matrix_dirs = ["code_count", "css_token", "domain_count", "fext_count",
                   "fn_count", "js_token", "space_count", "tag_token", 
                   "text_token", "url_count", "url_param_count", "whitespace_count",
                   "wordlen_raw_count"]
    feature_names = list()
    for mat_dir in matrix_dirs:
        train_mat_fn = os.path.join("matrix", mat_dir, "X_train.np")
        test_mat_fn = os.path.join("matrix", mat_dir, "X_test.np")
        train_df = sum_max_feat(train_df, train_mat_fn, mat_dir)
        test_df = sum_max_feat(test_df, test_mat_fn, mat_dir)
        feature_names.extend(["{}_sum".format(mat_dir), "{}_max".format(mat_dir)])
    
    # create directory if it does not exist
    if not os.path.isdir("matrix/sum_max_feat"):
        try:
            os.makedirs("matrix/sum_max_feat")
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    with open("matrix/sum_max_feat/X_train.np", "w") as fhandle:
        save_sparse_csr(fhandle, sparse.csr_matrix(train_df[feature_names].values))

    with open("matrix/sum_max_feat/X_test.np", "w") as fhandle:
        save_sparse_csr(fhandle, sparse.csr_matrix(test_df[feature_names].values))
