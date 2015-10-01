import string
import argparse
import urlparse
from datetime import datetime
from collections import Counter

import joblib


def get_domain(url):
    for token in url.split("/"):
        if token and "http" not in token:
            return token.replace("www", "").strip(string.punctuation).lower()


def derive_url_map():
    id_url_count_map = joblib.load("mapping/id_url_count_map.pkl")
    id_domain_count_map, id_url_param_count_map = dict(), dict()
    for _id in id_url_count_map:
        domains = list()
        parameters = list()
        for url,count in id_url_count_map[_id].items():
            domains.extend(count*[get_domain(url)])
            try:
                params = urlparse.parse_qs(urlparse.urlparse(url).query)
                for key in params:
                    for v in params[key]:
                        parameters.extend(count*["{}={}".format(key, v).lower()])
            except:
                pass

        id_domain_count_map[_id] = Counter(domains)
        id_url_param_count_map[_id] = Counter(parameters)

    # save derived mapping
    joblib.dump(id_domain_count_map, "mapping/id_domain_count_map.pkl")
    joblib.dump(id_url_param_count_map, "mapping/id_url_param_count_map.pkl")


def derive_fn_map():
    id_fn_count_map = joblib.load("mapping/id_fn_count_map.pkl")
    id_file_extension_count_map = dict()
    for _id in id_fn_count_map:
        extensions = list()
        for key,value in id_fn_count_map[_id].items():
            ext = key.split(".")[-1]
            extensions.extend(value*[ext])
        id_file_extension_count_map[_id] = Counter(extensions)

    # save derived mapping
    joblib.dump(id_file_extension_count_map, "mapping/id_file_extension_count_map.pkl")


# main function
def main():

   # parse the command-line argument
    parser = argparse.ArgumentParser(description="Extract feature matrices.")
    parser.add_argument("-f", "--func", help="specify derive mapping function", 
            required=True)

    # get the parameters
    args = vars(parser.parse_args())
    
    # functions used for feature extraction
    parse_func_map = {"derive_url_map":derive_url_map, "derive_fn_map":derive_fn_map}

    # process feature extraction & save data
    start = datetime.now()
    parse_func_map[args["func"]]()
    print "Elapsed time: {}".format(str(datetime.now() - start))


if __name__ == "__main__":
    main()
