import os
import re
import glob
import string
import multiprocessing
from collections import Counter, defaultdict

from bs4 import BeautifulSoup


# functions to repair broken html
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


# separate file list into fixed number of chunks
def chunks(seq, num):
    average, last = len(seq)/float(num), 0.0
    while last < len(seq):
        yield seq[int(last):int(last + average)]
        last += average
        

# repair the html files and output to repair directory
def repair_html(files):
    for f in files:
        filename = os.path.basename(f)
        html = open(f).read()
        html = auto_repair(html)
        try:
            html = BeautifulSoup(html.decode("utf-8", "ignore"), "lxml").prettify()
        except Exception as e:
            print "Error: {:<30}, File: {}".format(e, f)
        
        out_filename = os.path.join("repair/", filename)
        with open(out_filename, "w") as fhandle:
            fhandle.write(html.encode("utf-8"))


if __name__ == "__main__":

    # run the process in parallel
    num_process = multiprocessing.cpu_count()
    arg_list = [files for files in chunks(glob.glob("data/*/*"), num_process)]
    p = multiprocessing.Pool(num_process)
    p.map(repair_html, arg_list)
