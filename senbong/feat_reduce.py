import os
import argparse
from datetime import datetime

import numpy as np
import scipy.sparse as sparse


def feature_reduction(X, low_thresh=0.0, high_thresh=1.0):
    non_zero_mat = X > 0
    weight = non_zero_mat.sum(axis=0)/float(non_zero_mat.shape[0])
    condition = (low_thresh < weight) & (weight <= high_thresh)
    indices = np.where(np.array(condition).flatten())[0]
    return indices


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)

    
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                      shape=loader['shape'])


# main function
def main():

   # parse the command-line argument
    parser = argparse.ArgumentParser(description="Feature reduction")
    parser.add_argument("-i", "--input", help="specify input directory",
            required=True)
    parser.add_argument("-o", "--output", help="specify output directory", 
            required=True)
    parser.add_argument("-l", "--lower_thresh", help="specify lower threshold", 
            type=float, default=0.0)
    parser.add_argument("-u", "--upper_thresh", help="specify upper threshold", 
            type=float, default=1.0)

    # get the parameters
    args = vars(parser.parse_args())

    # perform feature reduction and save data
    start = datetime.now()
    with open(os.path.join(args["input"], "X_train.np")) as fhandle:
        X_train = load_sparse_csr(fhandle)

    with open(os.path.join(args["input"], "X_test.np")) as fhandle:
        X_test = load_sparse_csr(fhandle)
        
    indices = feature_reduction(X_train, low_thresh=args["lower_thresh"], 
            high_thresh=args["upper_thresh"])
    X_train_reduce = X_train[:,indices]
    X_test_reduce = X_test[:,indices]
    
    # write to output file
    with open(os.path.join(args["output"], "X_train_reduce.np"), "w") as fout:
        save_sparse_csr(fout, X_train_reduce)

    with open(os.path.join(args["output"], "X_test_reduce.np"), "w") as fout:
        save_sparse_csr(fout, X_test_reduce)

    print "Feature number from {:<10} to {}".format(X_train.shape[1], X_train_reduce.shape[1])
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
