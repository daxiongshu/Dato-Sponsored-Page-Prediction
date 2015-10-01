import os
import re
import glob
import json
import string
import argparse
import multiprocessing
from datetime import datetime
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from bs4 import BeautifulSoup


# useful regex
url = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.IGNORECASE)

file_extension = [ext.strip() for ext in open("data/file_extension.txt").readlines()]
pattern = r"(?<=[\"/])[\w.-]+?\.(?:{})(?=[\";'?#\)])".format("|".join(file_extension))
file_ext = re.compile(pattern)

# check code segment
segment_check = lambda x:"=" in x and ";" in x and "<" not in x and ">" not in x

# construct id to full path map
id_path_map = dict()


def chunks(seq, num):
    average, last = len(seq)/float(num), 0.0
    while last < len(seq):
        yield seq[int(last):int(last + average)]
        last += average


def extract_code_line(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_partial{}.pkl".format(process_no))
    id_code_count_map = dict()
    for i,f in enumerate(files):
        file_id = os.path.basename(f)
        codes = list()
        for line in open(f).readlines():
            if segment_check(line):
                codes.append(line.strip())
        id_code_count_map[file_id] = Counter(codes)
    # save to file
    joblib.dump(id_code_count_map, filename)
    
    
def visible(element):
    if element.parent.name in ["style", "script", "[document]", "head", "title"]:
        return False
    return True
    
    
def extract_text(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_chunk{}.json".format(process_no))
    with open(filename, "w") as fhandle:
        for f in files:
            extract_row = {"file_id": os.path.basename(f)}
            extract_row["title"] = ""
            extract_row["visible_text"] = ""
            try:
                soup = BeautifulSoup(open(f), "lxml")
                titles = " ".join([tag.get_text().strip() for tag in soup.find_all("title")])
                texts = soup.find_all(text=True)
                visible_texts = [text.strip() for text in texts if visible(text)]
                extract_row["title"] = titles
                extract_row["visible_text"] = " ".join(visible_texts)
            except Exception as e:
                print "Error: {}\tFile: {}".format(e, f)
                pass
            fhandle.write("{}\n".format(json.dumps(extract_row)))
            
        
def extract_url_fn(args):
    files, path, process_no = args
    url_filename = os.path.join(path, "id_url_partial{}_map.pkl".format(process_no))
    fn_filename = os.path.join(path, "id_fn_partial{}_map.pkl".format(process_no))

    id_url_count_map = dict()
    id_filename_count_map = dict()
    for i,f in enumerate(files):
        filename = os.path.basename(f)
        html = open(f).read()
        urls = url.findall(html)
        web_files = file_ext.findall(html)
        
        id_url_count_map[filename] = Counter(urls)
        id_filename_count_map[filename] = Counter(web_files) 
        
    # save the file
    joblib.dump(id_url_count_map, url_filename)
    joblib.dump(id_filename_count_map, fn_filename)
    
    
def extract_count_feat(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_partial{}.pkl".format(process_no))
    id_feat_count_map = dict()
    for i,f in enumerate(files):
        extract_data = dict()
        file_id = os.path.basename(f)
        with open(f) as fhandle:
            text = fhandle.read()
            extract_data["words"] = len(re.split("\s+", text))
            special_chars = 0
            for c in string.punctuation:
                extract_data[c] = text.count(c)
                special_chars += extract_data[c]
            extract_data["special_chars"] = special_chars

            int_link, ext_link, anchor = 0, 0, 0
            try:
                soup = BeautifulSoup(text, "lxml")
                for link in soup.find_all("a"):
                    if link.get("href") is not None:
                        href = link.get("href")
                        if "http" in href or "www" in href:
                            ext_link += 1
                        elif "#" in href and ("/" not in href or "." not in href):
                            anchor += 1
                        else:
                            int_link += 1
            except Exception as e:
                print "Error: {}\tFile: {}".format(e, f)

            # store link related features
            extract_data["int_link"] = int_link
            extract_data["anchor"] = anchor
            extract_data["ext_link"] = ext_link
            extract_data["int_link_ratio"] = int_link/float(int_link + ext_link + anchor + 1)
            extract_data["anchor_ratio"] = anchor/float(int_link + ext_link + anchor + 1)
            extract_data["ext_link_ratio"] = ext_link/float(int_link + ext_link + anchor + 1)

        # save to dict
        id_feat_count_map[file_id] = extract_data

    # save the file
    joblib.dump(id_feat_count_map, filename)


def extract_count_whitespace(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_partial{}.pkl".format(process_no))
    id_count_map = dict()
    for i,f in enumerate(files):
        extract_data = dict()
        file_id = os.path.basename(f)
        with open(f) as fhandle:
            text = fhandle.read()
            whitespace_chars = 0
            for c in string.whitespace:
                extract_data[c] = text.count(c)
                whitespace_chars += extract_data[c]
            extract_data["whitespace_chars"] = whitespace_chars

        # save to dict
        id_count_map[file_id] = extract_data

    # save the file
    joblib.dump(id_count_map, filename)


def extract_count_word_len(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_partial{}.pkl".format(process_no))
    id_count_map = dict()
    for i,f in enumerate(files):
        extract_data = dict()
        file_id = os.path.basename(f)
        with open(f) as fhandle:
            text = fhandle.read()
            id_count_map[file_id] = Counter(len(token) for token in text.split())

    # save the file
    joblib.dump(id_count_map, filename)
    
    
def extract_script(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_chunk{}.json".format(process_no))
    with open(filename, "w") as fhandle:
        for i,f in enumerate(files):
            extract_row = {"file_id": os.path.basename(f)}
            textdata = [""]
            try:
                soup = BeautifulSoup(open(f), "lxml")
                for text in soup.find_all("script"):
                    try:
                        textdata.append(text.text.encode('ascii','ignore').strip())
                    except Exception:
                        continue
            except Exception as e:
                print "Error: {}\tFile: {}".format(e, f)
            extract_row["text"] = filter(None, textdata)
            fhandle.write("{}\n".format(json.dumps(extract_row)))
            
            
def extract_style(args):
    files, path, process_no = args
    filename = os.path.join(path, "id_data_chunk{}.json".format(process_no))
    with open(filename, "w") as fhandle:
        for i,f in enumerate(files):
            extract_row = {"file_id": os.path.basename(f)}
            style_attr = list()
            style = list()
            
            try:
                soup = BeautifulSoup(open(f), "lxml")
                data_attribute_map = defaultdict(list)
                for tag in soup.findAll():
                    if "style" in tag.attrs:
                        style_attr.append((tag.name, tag.attrs))
                    if tag.name == "style":
                        style.append(tag.text.strip())
            except Exception as e:
                print "Error: {}\tFile: {}".format(e, f)
            extract_row["style_attribute"] = style_attr
            extract_row["style"] = style
            fhandle.write("{}\n".format(json.dumps(extract_row)))


# main function
def main():

    # parse the command-line argument
    parser = argparse.ArgumentParser(description="Extract feature to dictionary or json files.")
    parser.add_argument("-o", "--output", help="specify output directory",
            required=True)
    parser.add_argument("-f", "--func", help="specify function for data extraction", 
            required=True)
    parser.add_argument("--repair", help="use repaired html files",
            dest="repair", action="store_true")
    parser.add_argument("--no-repair", help="use original html files",
            dest="repair", action="store_false")
    parser.set_defaults(repair=True)
    parser.set_defaults(letter_case=True)

    # get the parameters
    args = vars(parser.parse_args())
    
    # set the id_path_map
    global id_path_map
    if args["repair"]:
        for f in glob.glob("repair/*"):
            file_id = os.path.basename(f)
            id_path_map[file_id] = f
    else:
        for f in glob.glob("data/*/*"):
            file_id = os.path.basename(f)
            id_path_map[file_id] = f

    # extract wanted files
    file_list = list()
    train_df = pd.read_csv("data/train_v2.csv")
    for _,row in train_df.iterrows():
        file_list.append(id_path_map[row["file"]])
        
    test_df = pd.read_csv("data/sampleSubmission_v2.csv")
    for _,row in test_df.iterrows():
        file_list.append(id_path_map[row["file"]])

    # functions used for feature extraction
    func_map = {"extract_code_line":extract_code_line, "extract_text":extract_text, 
            "extract_url_fn":extract_url_fn, "extract_count_feat":extract_count_feat,
            "extract_script":extract_script, "extract_style":extract_style,
            "extract_count_whitespace":extract_count_whitespace, 
            "extract_count_word_len":extract_count_word_len}

    # create directory if it does not exist
    if not os.path.isdir(args["output"]):
        try:
            os.makedirs(args["output"])
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise

    # prepare argument list
    num_process = multiprocessing.cpu_count()
    arg_list = [(files, args["output"], i) for i,files in enumerate(chunks(file_list, num_process))]
    start = datetime.now()
    p = multiprocessing.Pool(num_process)
    p.map(func_map[args["func"]], arg_list)
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
