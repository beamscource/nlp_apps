

import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

def extract_data(source_location, file_suffix, model):
    pass

def main(args):

    source_location = args.youtube_url
    model = args.embedding_model

    if os.path.isfile(source_location):
        file_suffix = file.split('.')[1]
        if file_suffix in ['pdf', 'xlxs']:
            data_frame = extract_data(source_location, file_suffix, model)
        print('No compatible file formats found.')

    elif os.path.isdir(source_location, file_suffix):

        for file in os.listdir(source_location):
            file_suffix = file.split('.')[1]

            if file_suffix in ['pdf', 'xlxs']:
                data_frame = extract_data(source_location, file_suffix, model)
                # concatenate data frames into a single one
            print('No compatible file formats found.')
    else:
        print('Given file/directory not found.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transcribe YT videos und \
    summarize.')
    parser.add_argument('-s', '--source_location', type=str, required=True, \
        help='Location of the files containing data.')
    parser.add_argument('-m', '--embedding_model', type=str, \
        default='', \
        help='HuggingFace model used for embedding the data.')
    main(parser.parse_args())