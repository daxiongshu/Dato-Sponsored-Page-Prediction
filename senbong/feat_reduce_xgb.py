import os
import glob
import argparse
from datetime import datetime

import numpy as np
import scipy.sparse as sparse


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)

    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                      shape=loader['shape'])


def get_feature_indices(path):
    indices = set()
    for f in glob.glob(os.path.join(path, "fscore*.txt")):
        feat_importances = eval(open(f).read())
        indices.update(feat_importances.keys())
        print "file: {:<50} feature no: {}".format(f, len(feat_importances))
    indices = [int(feat[1:]) for feat in indices]
    return indices


# main function
def main():

   # parse the command-line argument
    parser = argparse.ArgumentParser(description="Feature reduction")
    parser.add_argument("-i", "--input", help="specify input directory",
            required=True)
    parser.add_argument("-o", "--output", help="specify output directory", 
            required=True)

    # get the parameters
    args = vars(parser.parse_args())

    # perform feature reduction and save data
    start = datetime.now()
    with open(os.path.join(args["input"], "X_train.np")) as fhandle:
        X_train = load_sparse_csr(fhandle)

    with open(os.path.join(args["input"], "X_test.np")) as fhandle:
        X_test = load_sparse_csr(fhandle)
        
    indices = get_feature_indices(args["input"])
    X_train_reduce = X_train[:,indices]
    X_test_reduce = X_test[:,indices]
    
#    # write to output file
#    with open(os.path.join(args["output"], "X_train_reduce_xgb.np"), "w") as fout:
#        save_sparse_csr(fout, X_train_reduce)
#
#    with open(os.path.join(args["output"], "X_test_reduce_xgb.np"), "w") as fout:
#        save_sparse_csr(fout, X_test_reduce)

    print "Feature number from {:<10} to {}".format(X_train.shape[1], X_train_reduce.shape[1])
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
