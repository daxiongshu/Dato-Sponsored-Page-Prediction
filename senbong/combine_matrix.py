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

def combine_matrix(matrix_filenames):
    matrices = list()
    for matrix_fn in matrix_filenames:
        with open(matrix_fn) as fhandle:
            matrices.append(load_sparse_csr(fhandle))
    return sparse.hstack(matrices, format="csr")


if __name__ == "__main__":

    # parse the command-line argument
    parser = argparse.ArgumentParser(description="Combine feature matrices.")
    parser.add_argument("--reduce", help="use reduced feature matrices",
            dest="reduce", action="store_true")
    parser.add_argument("--no-reduce", help="use original feature matrices",
            dest="reduce", action="store_false")

    # get the parameters
    args = vars(parser.parse_args())

    # perform the processing
    start = datetime.now()
    matrix_dirs = list()
    for d in glob.glob("matrix/*"):
        if os.path.isdir(d):
            matrix_dirs.append(d)

    train_matrix_files, test_matrix_files = list(), list()
    for matrix_fn in matrix_dirs:
        if args["reduce"] and os.path.isfile(os.path.join(matrix_fn, "X_train_reduce.np")):
            train_matrix_files.append(os.path.join(matrix_fn, "X_train_reduce.np"))
        else:
            train_matrix_files.append(os.path.join(matrix_fn, "X_train.np"))

        if args["reduce"] and os.path.isfile(os.path.join(matrix_fn, "X_test_reduce.np")):
            test_matrix_files.append(os.path.join(matrix_fn, "X_test_reduce.np"))
        else:
            test_matrix_files.append(os.path.join(matrix_fn, "X_test.np"))

    X_train = combine_matrix(train_matrix_files)
    X_test = combine_matrix(test_matrix_files)

    train_filename = "matrix/X_train_reduce.np" if args["reduce"] else  "matrix/X_train_full.np"
    test_filename = "matrix/X_test_reduce.np" if args["reduce"] else  "matrix/X_test_full.np"
    with open(train_filename, "w") as fhandle:
        save_sparse_csr(fhandle, X_train)

    with open(test_filename, "w") as fhandle:
        save_sparse_csr(fhandle, X_test)

    print "Elapsed time: {}".format(str(datetime.now() - start))
