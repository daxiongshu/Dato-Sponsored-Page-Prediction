import os
import re
import glob
import string
import argparse
import multiprocessing
from datetime import datetime
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

import nltk
from nltk.tokenize import RegexpTokenizer


# useful regex
delimiter = re.compile("[\s;]+")
unwanted_chars = string.punctuation + string.whitespace

# construct id to full path map
id_path_map = dict()
train_files, test_files = list(), list()


# convert to str
def to_str(unicode_or_str):
    if isinstance(unicode_or_str, unicode):
        return unicode_or_str.encode("utf-8")
    else:
        return unicode_or_str


# generator to roll the list
def rolling(iterable, size, insufficient=True):
    len_func = (lambda x:max(x, 1)) if insufficient else lambda x:x
    for i in range(len_func(len(iterable) - size + 1)):
        yield iterable[i:i+size]


# functions to repair texts
def remove_spaces_between_chars(s, num_space):
    return re.sub(r"(?<=\w)\s{%s}(?=\w)" % num_space, "", s)


def remove_fix_spaces(s, num_space):
    return re.sub(r"\s{%s}" % num_space, "", s)


def shrink_spaces(s):
    return re.sub(r"\s{2,}", " ", s)


def auto_repair(s, char_threshold=1):
    spaces = re.split("\S", s)
    words = s.split()
    space_counter = Counter(len(sp) for sp in spaces)
    space_length = space_counter.most_common(1)[0][0]
    if space_length > 1:
        return shrink_spaces(remove_fix_spaces(s, space_length))
    if len(spaces) > len(words):
        return shrink_spaces(remove_spaces_between_chars(s, space_length))
    return shrink_spaces(s)


def sanitize_text(text):
    stemmer = nltk.stem.SnowballStemmer("english")
    tokenizer = RegexpTokenizer("[\w\"'-]+")
    return " ".join(stemmer.stem(token.lower()).rstrip("-")
                    for token in tokenizer.tokenize(text.decode("utf-8")) 
                    if token[0] not in string.punctuation and token.lower() not in ENGLISH_STOP_WORDS).encode("utf-8")


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    return True

    
# transformation functions
def transform_tav(tag_name, k, v, max_len=50, min_len=2):
    try:
        tag_name, k = tag_name.strip().lower(), k.strip().lower()
        v_str = v.strip(unwanted_chars).lower()
        if v_str and v_str not in ENGLISH_STOP_WORDS and len(v_str) < max_len and len(v_str) > min_len:
            return "{}.{}={}".format(to_str(tag_name), to_str(k), to_str(v_str))
    except Exception as error:
        print error
    
    
def transform_ta(tag_name, attribute):
    tag_name, attribute = tag_name.strip().lower(), attribute.strip().lower()
    return "{}.{}".format(to_str(tag_name), to_str(attribute))
    
    
def transform_av(attribute, value, max_len=50, min_len=2):
    try:
        attribute = attribute.strip().lower()
        v_str = value.strip(unwanted_chars).lower()
        if v_str and v_str not in ENGLISH_STOP_WORDS and len(v_str) < max_len and len(v_str) > min_len:
            return "{}={}".format(to_str(attribute), to_str(v_str))
    except Exception as error:
        print error
    
    
# parse file into tokens
def parse_all(f):
    tags = list()
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        for tag in soup.findAll():
            for k,v in tag.attrs.items():
                
                # add tag and attribute features
                tags.append(tag.name.strip().lower())
                tags.append(transform_ta(tag.name, k))
                
                if isinstance(v, list):
                    for _v in v:
                        tags.append(transform_tav(tag.name, k, _v))
                        tags.append(transform_av(k, _v))

                if isinstance(v, str):
                    if delimiter.search(v) and "http" not in v:
                        for _v in delimiter.split(v):
                            tags.append(transform_tav(tag.name, k, _v))
                            tags.append(transform_av(k, _v))
                    elif v:
                        tags.append(transform_tav(tag.name, k, v))
                        tags.append(transform_av(k, v))
        return " ".join(filter(None, tags))
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return " ".join(filter(None, tags))


# parse file into tokens: tag-attribute-value
def parse_tav(f):
    tags = list()
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        for tag in soup.findAll():
            for k,v in tag.attrs.items():
                if isinstance(v, list):
                    for _v in v:
                        tags.append(transform_tav(tag.name, k, _v))
        
                if isinstance(v, str):
                    if delimiter.search(v) and "http" not in v:
                        for _v in delimiter.split(v):
                            tags.append(transform_tav(tag.name, k, _v))
                    elif v:
                        tags.append(transform_tav(tag.name, k, v))
        return " ".join(filter(None, tags))
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return " ".join(filter(None, tags))


# parse file into tokens: tag-attribute
def parse_ta(f):
    tags = list()
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        for tag in soup.findAll():
            for k,v in tag.attrs.items():
                
                # add tag and attribute features
                tags.append(transform_ta(tag.name, k))
                
                if isinstance(v, list):
                    for _v in v:
                        tags.append(transform_av(k, _v))

                if isinstance(v, str):
                    if delimiter.search(v) and "http" not in v:
                        for _v in delimiter.split(v):
                            tags.append(transform_av(k, _v))
                    elif v:
                        tags.append(transform_av(k, v))
        return " ".join(filter(None, tags))
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return " ".join(filter(None, tags))


# parse file into tokens: tag-tag
def parse_tt(f):
    tags = list()
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        for pair in rolling([tag.name.strip() for tag in soup.findAll()], size=2, insufficient=False):
            tags.append(to_str("{}^{}".format(*pair)))
        return " ".join(filter(None, tags))
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return " ".join(filter(None, tags))


# parse file into tag: tag
def parse_t(f):
    tags = list()
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        for tag in soup.findAll():
            tags.append(to_str(tag.name.strip()))
        return " ".join(filter(None, tags))
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return " ".join(filter(None, tags))


# parse file into tokens: visible_text
def parse_text(f):
    try:
        html = open(f).read()
        soup = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml")
        texts = soup.findAll(text=True)
        visible_texts = [text for text in texts if visible(text)]
        visible_texts = [sanitize_text(auto_repair(text.encode("utf-8"))) for text in visible_texts]
        return " ".join(visible_texts).strip()
    except Exception as error:
        print "Error ({}): {}".format(error, f)
        return ""


# data stream generator
class DataGenerator(object):
    
    def __init__(self, files, parse_func, stop_words=None, num_process=-1):
        self.id_map = dict()
        self.files = files
        self.cur_chunk = None
        self.parse_func = parse_func
        if num_process == -1:
            self.num_process = multiprocessing.cpu_count()
        else:
            self.num_process = num_process
            
    def chunks(self):
        for i in xrange(0, len(self.files), self.num_process):
            yield self.files[i:i+self.num_process]
    
    def __iter__(self):
        id_count = 0
        p = multiprocessing.Pool(self.num_process)
        for files in self.chunks():
            self.cur_chunk = files
            for f,tokens in zip(files, p.map(self.parse_func, files)):
                filename = os.path.basename(f)
                self.id_map[filename] = id_count
                id_count += 1
                yield tokens


# prepare training & testing data
def feature_extract(parse_func, min_df, max_df, a, b, lowercase): 
    global train_files, test_files
    train_data_generator = DataGenerator(train_files, parse_func)
    test_data_generator = DataGenerator(test_files, parse_func)
    vectorizer = CountVectorizer(tokenizer=lambda x:x.split(), min_df=min_df, max_df=max_df,
            ngram_range=(a,b), lowercase=lowercase)
    X_train = vectorizer.fit_transform(train_data_generator)
    X_test = vectorizer.transform(test_data_generator)
    return X_train, X_test, vectorizer.vocabulary_


# save sparse matrix
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, 
             indptr=array.indptr, shape=array.shape)


# load sparse matrix
def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']), 
                      shape=loader["shape"])


# save extracted data
def save_data(X_train, X_test, vocabulary, path):

    filename = os.path.join(path, "X_train.np")
    with open(filename, "w") as fhandle:
        save_sparse_csr(fhandle, X_train)
    
    filename = os.path.join(path, "X_test.np")
    with open(filename, "w") as fhandle:
        save_sparse_csr(fhandle, X_test)

    filename = os.path.join(path, "vocabulary.pkl")
    joblib.dump(vocabulary, filename)


# main function
def main():

    # parse the command-line argument
    parser = argparse.ArgumentParser(description="Extract feature matrices.")
    parser.add_argument("-o", "--output", help="specify output directory",
            required=True)
    parser.add_argument("-p", "--parse_func", help="specify function for parsing", 
            required=True)
    parser.add_argument("-l", "--lower_thresh", help="specify min_df for CountVectorizer", 
            type=float, default=0.0)
    parser.add_argument("-u", "--upper_thresh", help="specify max_df for CountVectorizer", 
            type=float, default=1.0)
    parser.add_argument("-a", "--min_ngram", help="specify start ngram for CountVectorizer", 
            type=int, default=1)
    parser.add_argument("-b", "--max_ngram", help="specify end ngram for CountVectorizer", 
            type=int, default=1)
    parser.add_argument("--repair", help="use repaired html files",
            dest="repair", action="store_true")
    parser.add_argument("--no-repair", help="use original html files",
            dest="repair", action="store_false")
    parser.add_argument("--lower-case", help="lower lettercase for CountVectorizer",
            dest="letter_case", action="store_true")
    parser.add_argument("--no-lower-case", help="keep original lettercase for CountVectorizer",
            dest="letter_case", action="store_false")
    parser.set_defaults(repair=True)
    parser.set_defaults(letter_case=True)

    # get the parameters
    args = vars(parser.parse_args())
    args["max_ngram"] = max(args["min_ngram"], args["max_ngram"])
    
    # set the id_path_map
    global id_path_map, train_files, test_files
    if args["repair"]:
        for f in glob.glob("repair/*"):
            file_id = os.path.basename(f)
            id_path_map[file_id] = f
    else:
        for f in glob.glob("data/*/*"):
            file_id = os.path.basename(f)
            id_path_map[file_id] = f

    # extract wanted files
    train_df = pd.read_csv("data/train_v2.csv")
    for _,row in train_df.iterrows():
        train_files.append(id_path_map[row["file"]])
        
    test_df = pd.read_csv("data/sampleSubmission_v2.csv")
    for _,row in test_df.iterrows():
        test_files.append(id_path_map[row["file"]])

    # functions used for feature extraction
    parse_func_map = {"parse_all": parse_all, "parse_tav": parse_tav, "parse_ta": parse_ta,
                    "parse_text":parse_text, "parse_tt":parse_tt, "parse_t":parse_t}

    # create directory if it does not exist
    if not os.path.isdir(args["output"]):
        try:
            os.makedirs(args["output"])
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    # process feature extraction & save data
    start = datetime.now()
    X_train, X_test, vocabulary = feature_extract(parse_func_map[args["parse_func"]], 
            args["lower_thresh"], args["upper_thresh"], args["min_ngram"], args["max_ngram"],
            args["letter_case"])
    save_data(X_train, X_test, vocabulary, args["output"])
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
