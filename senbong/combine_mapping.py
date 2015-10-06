import glob
import argparse

import joblib


def combine_mapping(pattern, output_fn):
    id_data_map = dict()
    for f in glob.glob(pattern):
        id_data_map.update(joblib.load(f))
    joblib.dump(id_data_map, output_fn)


def main():

    # parse the command-line argument
    parser = argparse.ArgumentParser(description="Combine mapping files.")
    parser.add_argument("-o", "--output", help="specify output filename",
            required=True)
    parser.add_argument("-p", "--pattern", help="specify input file pattern", 
            required=True)

    # get the parameters
    args = vars(parser.parse_args())
    combine_mapping(args["pattern"], args["output"])


if __name__ == "__main__":
    main()
