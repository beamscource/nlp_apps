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

import argparse
import os
from math import floor
from tqdm import tqdm
import time
import datetime

from database_handling import connect_mysql_sql_db, download_all_articles_by_date

from aleph_alpha_client import AlephAlphaModel, AsyncClient, Prompt, CompletionRequest
from few_shots_lib import topic_from_keys_shots, summary_shots_translation_shorter, \
    summary_shots

import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from visualize_topic_model import visualize_topic_model

def model_clusters(documents, plot_model, run_start, category_tag):

    vectorizer_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    #vectorizer_model = CountVectorizer(stop_words="english")
    #sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    sentence_model = SentenceTransformer('allenai-specter')

    # create documents embeddings
    print(f'=================================================================')
    print(f'Embedding documents with the tag {category_tag} as vectors.')

    embeddings = sentence_model.encode(documents, show_progress_bar=True)

    # define UMAP model to reduce embeddings dimension
    umap_model = umap.UMAP(n_neighbors=15,
                        n_components=10,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=False,
                        random_state=42)

    if len(documents) < 30:
        min_cluster_size = 4
        min_samples = 1
    else:
        min_cluster_size = 5
        min_samples = 3

    # Define HDBSCAN model to perform documents clustering
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    cluster_selection_epsilon=0.0,
                                    cluster_selection_method='leaf',
                                    prediction_data=True)

    topic_model = BERTopic(top_n_words=10,
                       vectorizer_model=vectorizer_model,
                       calculate_probabilities=False,
                       umap_model= umap_model,
                       hdbscan_model=hdbscan_model,
                       verbose=False)

    print(f'=================================================================')
    print(f'Computing the topic model.')
    topic_model.fit(documents, embeddings)

    print(f'=================================================================')
    print(f'Extracting representative terms for the clusters.')
    clusters_with_keywords = topic_model.topic_representations_

    top_clusters_with_terms = {}
    for cluster in clusters_with_keywords:
        top_clusters_with_terms[cluster] = [keyword[0] for keyword in \
                clusters_with_keywords[cluster]]

    if plot_model:
        run_dir = os.path.join('./_pipeline_runs/', run_start)
        if not os.path.isdir(run_dir):
            os.mkdir(run_dir)
        visualize_topic_model(topic_model, run_dir, category_tag)

    document_info = topic_model.get_document_info(documents)
    document_cluster_mapping = document_info[['Document', 'Topic',\
                                              'Representative_document']]

    breakpoint()
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

async def summarize_translate(abstracts, token):
    few_shots = summary_shots_translation_shorter()

    async with AsyncClient(token=token) as client:
        requests = (
            CompletionRequest(prompt=Prompt(few_shots.format(abstract)),
                              maximum_tokens=200, stop_sequences=['###'])
            for abstract in abstracts)

        results = await asyncio.gather(
            *(client.complete(request, model="luminous-extended")
              for request in requests)
            )
        await client.close()

    return results

def filter_abstracts_in_clusters(document_cluster_mapping, top_clusters_with_terms):

    document_cluster_mapping['Keyword_match'] = np.nan
    for cluster in top_clusters_with_terms:
        if cluster == -1:
            continue
        keywords_set = set(top_clusters_with_terms[cluster])
        abstracts = document_cluster_mapping[document_cluster_mapping\
                                ['Topic'] == cluster]['Document'].tolist()
        indeces = document_cluster_mapping[document_cluster_mapping\
                                ['Topic'] == cluster]['Document'].index.tolist()

        for abstract, index in zip(abstracts, indeces):
            abstract_wordset = set(abstract.lower().split())
            # check the intersection size between cluster keywords and the abstract
            if len(abstract_wordset.intersection(keywords_set)) >= \
                floor(len(keywords_set)/3):
                document_cluster_mapping.loc[index, 'Keyword_match'] = True
            else:
                document_cluster_mapping.loc[index, 'Keyword_match'] = False

    return document_cluster_mapping

def save_results(document_cluster_mapping_total, run_start):
    
    run_dir = os.path.join('./_pipeline_runs/', run_start)
    with pd.ExcelWriter(os.path.join(run_dir, '_general_results.xlsx'), \
                        engine='openpyxl', mode='w') as writer:
        document_cluster_mapping_total.to_excel(writer, \
                                        sheet_name=f'{run_start}_docs', \
                                        index=False)

def save_output_as_json(document_cluster_mapping_total, run_start_str):
    pass

def main(args):

    start_date = args.start_date
    end_date = args.end_date
    token_file = args.token_file
    plot_model = args.plot_model

    run_start = round(time.time())
    run_start_str = str(run_start)

    # get entries from the DB
    connection = connect_mysql_sql_db()
    db_entries = download_all_articles_by_date(connection, start_date, end_date)
    print(f'=================================================================')
    print(f'Fetched {len(db_entries)} entries from the DB.')
    # close the SQL connection
    connection.close()

    # loop over tags provided by arxive
    # Coputer Vision, language and NLP, Audio and Speech, Machine Learning
    relevant_tags = ['cs.CV', 'cs.CL', 'eess.AS', 'cs.LG']

    for index, category_tag in enumerate(tqdm(relevant_tags)):

        if category_tag == 'cs.LG':
            tagged_db_entries = db_entries[db_entries['categories']\
                                    .str.contains(category_tag, na=False)]
            relevant_tags_short = relevant_tags[:]
            relevant_tags_short.remove(category_tag)
            for tag in relevant_tags_short:
                tagged_db_entries = tagged_db_entries[~tagged_db_entries['categories']\
                                    .str.contains(tag, na=False)]
                #tagged_db_entries.reset_index(inplace=True)

        else:
            tagged_db_entries = db_entries[db_entries['categories']\
                                    .str.contains(category_tag, na=False)]
        tagged_db_entries.reset_index(inplace=True)

        # model clusters with UMAP, HDBSCAN, and BERTopic
        top_clusters_with_terms, document_cluster_mapping = \
            model_clusters(tagged_db_entries['summary'].str.replace('\n', ' ', regex=True),\
                           plot_model, run_start_str, category_tag)

        # add Arxive category tag
        document_cluster_mapping['Arxive_tag'] = category_tag

        # filter clustered abstracts based on cluster keywords
        document_cluster_mapping = filter_abstracts_in_clusters(document_cluster_mapping,\
                                                        top_clusters_with_terms)

        # label clusters with Aleph Alpha
        token = read_aa_token(token_file)
        labeled_clusters = label_clusters(top_clusters_with_terms, token)

        # get translated summary from Aleph Alpha
        abstracts = document_cluster_mapping['Document']
        results = summarize_translate(abstracts, token)

        translated_summaries = {}
        for result, abstract in zip(results, abstracts):
            translated_summaries[abstract] = \
                result.completions[0].completion.split('\n')[0].strip()

        # match abstracts with Aleph Alpha topic labels and summaries
        document_cluster_mapping['Topic_label'] = \
                        document_cluster_mapping['Topic'].map(labeled_clusters)
        document_cluster_mapping['Document_summary'] = \
                 document_cluster_mapping['Document'].map(translated_summaries)
        
        # add paper counts
        document_cluster_mapping['Paper_counts'] = document_cluster_mapping\
            .groupby(['Topic_label'])['Topic_label'].transform('count')
        
        # using keyword matches only
        document_cluster_mapping['Paper_counts_matches'] = document_cluster_mapping\
            [document_cluster_mapping['Keyword_match'] == True]\
            .groupby(['Topic_label'])['Topic_label'].transform('count')
        
        # match abstracts with full PDF links
        base_url = 'https://arxiv.org/abs/'
        tagged_db_entries['pdf_link'] = base_url + tagged_db_entries['filename']
        pdf_links = dict(zip(tagged_db_entries['summary'].str.replace('\n', ' ',\
                            regex=True), tagged_db_entries['pdf_link']))
        document_cluster_mapping['PDF_link'] = \
            document_cluster_mapping['Document'].map(pdf_links)

        # get rid of the Topic column
        document_cluster_mapping = document_cluster_mapping.drop('Topic', axis=1)
        
        # concatenate results for all arxive tags
        if index == 0:
            document_cluster_mapping_total = document_cluster_mapping
        else:
            document_cluster_mapping_total = pd.concat([document_cluster_mapping_total, \
                            document_cluster_mapping]).reset_index(drop=True)
        breakpoint()
    # output table for debugging
    save_results(document_cluster_mapping_total, run_start_str)

    # TO DO: save all results to a JSON file
    # for format see https://github.com/zalando/tech-radar/blob/master/docs/index.html
    # for format see https://github.com/thoughtworks/build-your-own-radar#using-csv-data
    save_output_as_json(document_cluster_mapping_total, run_start_str)

    t1 = time.time()
    print(f'=================================================================')
    print(f'The whole pipeline took {str(datetime.timedelta(seconds=round(t1-run_start)))} \
          to complete.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Topic modelling-Aleph Alpha \
            pipeline which downloads abstracts from a DB, modelling topic clusters,\
            adds titles to them and saves all results to JSON file.')

    parser.add_argument('-sd', '--start_date', type=str, \
            default='2022-01-01', help='Earliest publishing date.')
    parser.add_argument('-ed', '--end_date', type=str, \
            default='2022-01-31', help='Latest publishing date.')
    parser.add_argument('-t', '--token_file', type=str, \
            default='token', help='File where the Aleph Alpha token is stored.')
    parser.add_argument('-pm', '--plot_model', type=bool, \
            default=True, help='If set to True, topic model is visualized.')

    main(parser.parse_args())

