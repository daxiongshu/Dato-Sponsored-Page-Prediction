import os
import re
import glob
import json
import string
import argparse
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


# split the string into tokens
js_splitter = re.compile("[{}{}]".format(string.whitespace, string.punctuation))

# split the string into tokens
wanted_chars = ""
unwanted_chars = "".join([c for c in string.punctuation if c not in wanted_chars])
css_splitter = re.compile("[{}{}]".format(string.whitespace, unwanted_chars))

# id to index mapping for javascript and css
id_js_index_map = dict()
id_css_index_map = dict()
id_text_index_map = dict()

# data generators
def js_token_data_generator():
    global id_js_index_map
    for f in glob.glob("json/script/*.json"):
        for line in open(f).readlines():
            extract_row = json.loads(line)
            id_js_index_map[extract_row["file_id"]] = len(id_js_index_map)
            script = " ".join(extract_row["text"])
            tokens = [token for token in js_splitter.split(script)]            
            tokens = filter(None, tokens)
            yield " ".join(tokens)
            
            
def css_token_data_generator():
    global id_css_index_map
    for f in glob.glob("json/style/*.json"):
        for line in open(f).readlines():
            extract_row = json.loads(line)
            id_css_index_map[extract_row["file_id"]] = len(id_css_index_map)
            tokens = [token for token in css_splitter.split(" ".join(extract_row["style"])) if token]
            for data in extract_row["style_attribute"]:
                tokens.extend(css_splitter.split(data[1]["style"]))
            yield " ".join(tokens)


def text_token_data_generator():
    global id_text_index_map
    translation_table = string.maketrans(string.punctuation+string.uppercase,
                                            " "*len(string.punctuation)+string.lowercase)
    snowball_stemmer = SnowballStemmer("english")
    for f in glob.glob("json/text/*.json"):
        for line in open(f).readlines():
            extract_row = json.loads(line)
            id_text_index_map[extract_row["file_id"]] = len(id_text_index_map)
            visible_text = extract_row["visible_text"].encode("ascii", "ignore")
            visible_text = visible_text.translate(translation_table)
            visible_text = [snowball_stemmer.stem(word) for word in visible_text.split()
                            if word not in ENGLISH_STOP_WORDS and len(word) > 1]
            title = extract_row["title"].encode("ascii", "ignore")
            title = title.translate(translation_table)
            title = ["t^{}".format(snowball_stemmer.stem(word)) for word in title.split()
                     if word not in ENGLISH_STOP_WORDS and len(word) > 1]
            visible_text.extend(title)
            yield " ".join(visible_text)


# save sparse matrix
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)


# functions to extract tokens and output matrix
def token_matrix(outdir, data_generator, map_func):

    # transform token data into matrix
    vectorizer = CountVectorizer(tokenizer=lambda x:x.split(), lowercase=False)
    X = vectorizer.fit_transform(data_generator())
    
    # extract indices
    train_df = pd.read_csv("data/train_v2.csv")
    test_df = pd.read_csv("data/sampleSubmission_v2.csv")
    train_idx = train_df["file"].apply(map_func).values
    test_idx = test_df["file"].apply(map_func).values

    # prepare X_train & X_test
    X_train, X_test = X[train_idx], X[test_idx]
    
    # create directory if it does not exist
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    # save matrices
    with open(os.path.join(outdir, "X_train.np"), "w") as fhandle:
        save_sparse_csr(fhandle, X_train)

    with open(os.path.join(outdir, "X_test.np"), "w") as fhandle:
        save_sparse_csr(fhandle, X_test)

    joblib.dump(vectorizer.vocabulary_, os.path.join(outdir, "vocabulary.pkl"))


# main function
def main():
   # parse the command-line argument
    parser = argparse.ArgumentParser(description="Extract feature matrices from json files.")
    parser.add_argument("-o", "--output", help="specify output directory",
            required=True)
    parser.add_argument("-f", "--func", help="specify function for extracting matrix", 
            required=True)

    # get the parameters
    args = vars(parser.parse_args())
    
    # functions used for feature extraction
    js_token_matrix = lambda x:token_matrix(x, js_token_data_generator, lambda x:id_js_index_map[x])
    css_token_matrix = lambda x:token_matrix(x, css_token_data_generator, lambda x:id_css_index_map[x])
    text_token_matrix = lambda x:token_matrix(x, text_token_data_generator, lambda x:id_text_index_map[x])
    func_map = {"js_token_matrix":js_token_matrix, "css_token_matrix":css_token_matrix, 
            "text_token_matrix":text_token_matrix}

    # process feature extraction & save data
    start = datetime.now()
    func_map[args["func"]](args["output"])
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
