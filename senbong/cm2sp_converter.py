import os
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse


class CountMapToSparseMat(object):
    
    def __init__(self, id_count_map):
        self.id_count_map = id_count_map
        
        all_features = set()
        feature_index_map = dict()
        for _id in id_count_map:
            all_features.update(id_count_map[_id])
            
        all_features = sorted(list(all_features))
        for i,f in enumerate(all_features):
            feature_index_map[f] = i
            
        self.all_features = all_features
        self.feature_index_map = feature_index_map
        
    
    def convert_to_sparse(self, df, id_field, id_func=lambda x:x):
        rows, cols, data = list(), list(), list()
        for i,(_,row) in enumerate(df.iterrows()):
            id_val = id_func(row[id_field])
            for f in self.id_count_map[id_val]:
                rows.append(i)
                cols.append(self.feature_index_map[f])
                data.append(self.id_count_map[id_val][f])
        return sparse.csr_matrix((data, (rows, cols)), shape=(df.shape[0], len(self.feature_index_map)))


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)


# main function
def main():

   # parse the command-line argument
    parser = argparse.ArgumentParser(description="Extract feature matrices.")
    parser.add_argument("-o", "--output", help="specify output directory",
            required=True)
    parser.add_argument("-i", "--input", help="specify input mapping file", 
            required=True)

    # get the parameters
    args = vars(parser.parse_args())
    
    # extract wanted files
    train_df = pd.read_csv("data/train_v2.csv")
    test_df = pd.read_csv("data/sampleSubmission_v2.csv")

    # create directory if it does not exist
    if not os.path.isdir(args["output"]):
        try:
            os.makedirs(args["output"])
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    
    # load count map dictionary
    id_data_count_map = joblib.load(args["input"])

    # process feature extraction & save data
    start = datetime.now()
    cm2sm_converter = CountMapToSparseMat(id_data_count_map)
    X_train = cm2sm_converter.convert_to_sparse(train_df, id_field="file")
    X_test = cm2sm_converter.convert_to_sparse(test_df, id_field="file")

    # save matrices
    with open(os.path.join(args["output"], "X_train.np"), "w") as fhandle:
        save_sparse_csr(fhandle, X_train)

    with open(os.path.join(args["output"], "X_test.np"), "w") as fhandle:
        save_sparse_csr(fhandle, X_test)
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
