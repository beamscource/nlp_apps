'''A pipeline script levereging topic modelling with BERTopic and an Aleph Alpha
    completion prompt to cluster and categorize research papers based on their
    abstract.

    NOTE: The pipeline assumes a SQL database which stores texts along with their
    publication dates. The logic to connect to the database is written in
    database_handling.py.
    
    Single steps include:
    1) Connecting to SQL database and fetching article based on their publication
    date.
    2) Using BERTopic to identify topic clusters and to extract top-rated keywords
    for each of them. It's possible to use a sparse or a dense model.
    3) Using an Aleph Alpha completion prompt with few shots to identify a general
    topic for each cluster.
    4) TO DO: save the results to a CSV file to pass the data to a D3-based frontend.
    For more info see https://github.com/thoughtworks/build-your-own-radar#using-csv-data
    
    To use the Aleph Alpha API, you have to save your token in the same directory
    as the ai_radar_pipeline.py script in a file called token (without any extension).
    
    Usage at the terminal:
    
    To see all currently implemented script parameters use
    python ai_radar_pipeline.py --help

    1) start the script with default parameters
    python ai_radar_pipeline.py

    2) change the time period of the publications to dowload all articles 
    published between January and December 2022
    python ai_radar_pipeline.py -sd '2022-01-01' -ed '2022-12-31'

    3) change the name of the token file
    python ai_radar_pipeline.py -t token_id

    The script uses few_shots_lib.py to load specific few shots which are then used
    with the Aleph Alpha completion prompt. To adapt your few shots, you should
    change or add a new function to the few_shots_lib.py

    '''
# general imports
import argparse
import os
from tqdm import tqdm
import time

# data base handling
from database_handling import connect_mysql_sql_db, download_all_articles_by_date

# aleph alpha
from aleph_alpha_client import AlephAlphaModel, Prompt, CompletionRequest
from few_shots_lib import topic_shots, topic_from_keys_shots

# data and topic modelling
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from visualize_topic_model import visualize_topic_model

def model_clusters(documents, plot_model):

    vectorizer_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    topic_model = BERTopic(vectorizer_model=vectorizer_model, \
                           embedding_model=sentence_model)
    print(f'=================================================================')
    print(f'Computing the topic model.')
    topic_model.fit(documents)

    print(f'=================================================================')
    print(f'Extracting representative terms for the clusters.')
    clusters_with_keywords = topic_model.topic_representations_

    top_clusters_with_terms = {}
    for cluster in clusters_with_keywords:
        top_clusters_with_terms[cluster] = [keyword[0] for keyword in \
                clusters_with_keywords[cluster]]

    if plot_model:
        visualize_topic_model(topic_model)
    
    document_info = topic_model.get_document_info(documents)
    document_cluster_mapping = document_info[['Document', 'Topic', 'Probability']]

    return top_clusters_with_terms, document_cluster_mapping

def read_aa_token(token_file):

    if os.path.isfile(token_file):
        with open(token_file, 'r', encoding='utf-8') as f:
            token = f.readlines()

    return token[0]

def label_clusters(top_clusters_with_terms, token):

    # for API parameters see
    # https://github.com/Aleph-Alpha/aleph-alpha-client/blob/main/aleph_alpha_client/completion.py#L6
    model = AlephAlphaModel.from_model_name(model_name="luminous-extended", token=token)
    few_shots = topic_from_keys_shots()

    labeled_clusters = {}
    print(f'=================================================================')
    print(f'Labelling topic clusters.')
    for cluster, keywords in tqdm(top_clusters_with_terms.items()):
        prompted_keywords = Prompt(few_shots.format(' '.join(keywords)))
        request = CompletionRequest(prompt=prompted_keywords, maximum_tokens=5, \
                                    stop_sequences=['###'])
        result = model.complete(request)
        labeled_clusters[cluster] = \
                        result.completions[0].completion.split('\n')[0].strip()

    return labeled_clusters

def main(args):

    start_date = args.start_date
    end_date = args.end_date
    token_file = args.token_file
    plot_model = args.plot_model

    # get entries from the DB
    connection = connect_mysql_sql_db()
    db_entries = download_all_articles_by_date(connection, start_date, end_date)
    print(f'=================================================================')
    print(f'Fetched {len(db_entries)} entries from the DB.')
    # close the SQL connection
    connection.close()

    # model with BERTopic
    top_clusters_with_terms, document_cluster_mapping = model_clusters(db_entries['summary'],\
                                                                          plot_model)

    # label_clusters with Aleph Alpha
    token = read_aa_token(token_file)
    labeled_clusters = label_clusters(top_clusters_with_terms, token)

    # match original documents with Aleph Alpha labels
    document_cluster_mapping['cluster_label'] = \
                        document_cluster_mapping['Topic'].map(labeled_clusters)
    document_cluster_mapping.columns = ['abstract', 'cluster', 'doc_probability',\
                                        'cluster_label']
    document_cluster_mapping.to_excel('aa_labeled_clusters.xlsx', \
                                      sheet_name='99_docs', index=False)
    # save all results to a CSV file
    # for format see https://github.com/thoughtworks/build-your-own-radar#using-csv-data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A BERT-Aleph Alpha pipeline \
            downloading abstracts from a DB, modelling topic clusters, and \
            adding titles to them.')

    parser.add_argument('-sd', '--start_date', type=str, \
            default='2022-01-20', help='Earliest publishing date.')
    parser.add_argument('-ed', '--end_date', type=str, \
            default='2022-01-22', help='Latest publishing date.')
    parser.add_argument('-t', '--token_file', type=str, \
            default='token', help='File where the Aleph Alpha token is stored.')
    parser.add_argument('-pm', '--plot_model', type=bool, \
            default=True, help='If set to True, topic model is visualized.')

    main(parser.parse_args())

