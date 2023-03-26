'''A pipeline script levereging topic modelling with BERTopic and an Aleph Alpha
    completion prompt to cluster and categorize research papers based on their
    abstract.

    NOTE: The pipeline assumes a SQL database which stores texts along with their
    publication dates. The logic to connect to the database is written in
    database_handling.py.
    
    Single steps include:
    1) Connecting to SQL database and fetching article based on their publication
    date. If manual_mode flag is set to true, dates are passed via the script
    parameters. Otherwise, the download dates are chosen in relation to the time
    stamp when the pipeline is run. This allows automatic deployment of the script.

    2) Using UMAP/HDBSCAN/BERTopic to identify topic clusters and to extract
    top-rated keywords for each of them. A SentenceTransformer model is used to
    embbed the incoming text.

    3) Using an Aleph Alpha completion prompt with few shots to identify a general
    topic label for each cluster.

    4) Using an Aleph Alpha completion prompt with few shots to summarize and
    translate each abstract.

    5) Combining processed data with metadata (Authors, Titles, PDF links).

    6) Converting and saving all results to a JSON file which will be consumed by
    a GUI. For more info see https://github.com/thoughtworks/build-your-own-radar
    When manual_mode is set to True additionaly saving the results to a .xlsx
    file for documentation/debugging purposes.
    
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
from collections import defaultdict
import json
import os
from math import floor
from tqdm import tqdm
import time
import datetime
from dateutil.relativedelta import relativedelta

from google.cloud import storage

import asyncio
from database_handling import connect_mysql_sql_db, download_all_articles_by_date

from aleph_alpha_client import AlephAlphaModel, AsyncClient, Prompt, CompletionRequest
from few_shots_lib import topic_from_keys_shots, summary_shots_translation_shorter, \
    summary_shots_translation_tiny, summary_shots

import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from visualize_topic_model import visualize_topic_model

def model_clusters(documents, manual_mode, run_start, category_tag):

    vectorizer_model = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    #vectorizer_model = CountVectorizer(stop_words="english")
    #sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    sentence_model = SentenceTransformer('allenai-specter')

    # create documents embeddings
    print(f'=================================================================')
    print(f'Embedding documents with the tag {category_tag} as vectors.')

    try:
        embeddings = sentence_model.encode(documents, show_progress_bar=True)
    except:
        print('GPUs RAM is probably full. Try to flush it first.')
        breakpoint()

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

    if manual_mode:
        run_dir = os.path.join('./_pipeline_runs/', run_start)
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir)
        visualize_topic_model(topic_model, run_dir, category_tag)

    document_info = topic_model.get_document_info(documents)
    document_cluster_mapping = document_info[['Document', 'Topic',\
                                              'Representative_document']]

    return top_clusters_with_terms, document_cluster_mapping

def read_aa_token(token_file):

    if os.path.isfile(token_file):
        with open(token_file, 'r', encoding='utf-8') as f:
            token = f.readlines()

    return token[0].strip()

def label_clusters(top_clusters_with_terms, token):

    # for API parameters see
    # https://github.com/Aleph-Alpha/aleph-alpha-client/blob/main/aleph_alpha_client/completion.py#L6
    model = AlephAlphaModel.from_model_name(model_name="luminous-extended", token=token)
    few_shots = topic_from_keys_shots()

    labeled_clusters = {}
    print(f'=================================================================')
    print(f'Labelling topic clusters.')
    for cluster, keywords in tqdm(top_clusters_with_terms.items()):
        prompted_keywords = Prompt.from_text(few_shots.format(' '.join(keywords)))
        request = CompletionRequest(prompt=prompted_keywords, maximum_tokens=5, \
                                    stop_sequences=['###'])
        result = model.complete(request)
        labeled_clusters[cluster] = \
                        result.completions[0].completion.split('\n')[0].strip()

    return labeled_clusters

async def summarize_translate(abstracts, token):
    few_shots = summary_shots_translation_tiny()
    few_shots = few_shots.strip().replace('\n   ', '')

    async with AsyncClient(token=token) as client:
        requests = (
            CompletionRequest(prompt=Prompt.from_text(few_shots.format(abstract)),
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

def save_results_as_xlsx(document_cluster_mapping_total, run_start):
    
    run_dir = os.path.join('./_pipeline_runs/', run_start)
    with pd.ExcelWriter(os.path.join(run_dir, '_general_results.xlsx'), \
                        engine='openpyxl', mode='w') as writer:
        document_cluster_mapping_total.to_excel(writer, \
                                        sheet_name=f'{run_start}_docs', \
                                        index=False)

def save_output_as_json(document_cluster_mapping, run_start_str):

    # convert data
    rings_category = pd.cut(document_cluster_mapping['Paper_counts'], \
                            bins=[0,6,12,25,120], labels=[3, 2, 1, 0])
    document_cluster_mapping.insert(5, 'Radar_ring', rings_category)

    # compile source data frame
    json_data_frame = pd.DataFrame(columns = ['name', 'ring', 'quadrant', \
                                'isNew', 'authors', 'title', 'summary', 'link'])
    json_data_frame['name'] = document_cluster_mapping['Topic_label'].str.capitalize()
    json_data_frame['ring'] = document_cluster_mapping['Radar_ring']\
        .apply(lambda x: 'adopt' if x <= 1 else 'hype')
    json_data_frame['quadrant'] = document_cluster_mapping['Arxive_tag']
    json_data_frame['isNew'] = 'FALSE'
    json_data_frame['authors'] = document_cluster_mapping['Authors']\
        .str.replace(';', ',', regex=True).str.strip()
    json_data_frame['title'] = document_cluster_mapping['Title']\
        .str.replace('\n', '', regex=True).str.replace('  ', ' ', regex=True).str.strip()
    json_data_frame['summary'] = document_cluster_mapping['Document_summary']
    json_data_frame['link'] = document_cluster_mapping['PDF_link']

    # build JSON
    topic_json_list = []

    for topic_cluster in json_data_frame['name'].unique():
        topic_json = defaultdict(list)
        sub_data = json_data_frame[json_data_frame['name'] == topic_cluster]
        topic_json["name"] = sub_data["name"].iloc[0]
        topic_json["ring"] = sub_data["ring"].iloc[0]
        if sub_data["quadrant"].iloc[0] == 'cs.LG':
            topic_json["quadrant"] = 'Machine learning'
        elif sub_data["quadrant"].iloc[0] == 'cs.CV':
            topic_json["quadrant"] = 'Computer Vision'
        elif sub_data["quadrant"].iloc[0] == 'cs.CL':
            topic_json["quadrant"] = 'Natural Language Processing'
        else:
            topic_json["quadrant"] = 'Audio and Speech Processing'
        topic_json["isNew"] = 'FALSE'

        # limit loop to 5 iterations = 5 papers
        for i, row in enumerate(sub_data[:5].iterrows()):
            #authors_formatted = f'<h5>{row[1].authors}</h5>'
            title_formatted = f'<h5>{row[1].title.strip()}</h5>'
            summary_formatted = f'<p>{row[1].summary.strip()}</p>'
            link_formatted = f'<a href=\"{row[1].link.strip()}\">{row[1].link}</a>'
            # combine into a single string
            if i == 0:
                description_formatted = title_formatted + summary_formatted + \
                    link_formatted
            else:
                description_formatted = description_formatted + title_formatted + \
                    summary_formatted + link_formatted
        topic_json["description"] = description_formatted
        topic_json_list.append(topic_json)

    json_object = json.dumps(topic_json_list, indent=4, ensure_ascii=False)
    breakpoint()

    with open(f'_pipeline_output_{run_start_str}.json', 'w') as outfile:
        outfile.write(json_object)


def main(args):

    manual_mode = args.manual_mode

    if manual_mode:
        start_date = args.start_date
        end_date = args.end_date
    else:
        # assuming the pipeline runs on the first of each month
        todays_date = datetime.datetime.today().strftime('%Y-%m-%d')
        # a month before (first of the previous month)
        start_date = datetime.datetime.strftime(datetime.datetime.strptime\
                    (todays_date, '%Y-%m-%d') - relativedelta(months=1), '%Y-%m-%d')
        # yesterday (last day of the previous month)
        end_date = datetime.datetime.strftime(datetime.datetime.strptime\
                    (todays_date, '%Y-%m-%d') - relativedelta(days=1), '%Y-%m-%d')

    token_file = args.token_file

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

        else:
            tagged_db_entries = db_entries[db_entries['categories']\
                                    .str.contains(category_tag, na=False)]
        tagged_db_entries.reset_index(inplace=True)

        # model clusters with UMAP, HDBSCAN, and BERTopic
        top_clusters_with_terms, document_cluster_mapping = \
            model_clusters(tagged_db_entries['summary'].str.replace('\n', ' ', regex=True),\
                           manual_mode, run_start_str, category_tag)

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
        results = asyncio.run(summarize_translate(abstracts, token))

        translated_summaries = {}
        for result, abstract in zip(results, abstracts):
            translated_summaries[abstract] = \
                result.completions[0].completion.split('\n')[0].strip()

        # match abstracts with Aleph Alpha topic labels and summaries
        document_cluster_mapping['Topic_label'] = \
                        document_cluster_mapping['Topic'].map(labeled_clusters)
        document_cluster_mapping['Document_summary'] = \
                 document_cluster_mapping['Document'].map(translated_summaries)

        # ADDING META INFORMATION TO THE TOPIC MODELLING RESULTS

        # add paper counts
        document_cluster_mapping['Paper_counts'] = document_cluster_mapping\
            .groupby(['Topic_label'])['Topic_label'].transform('count')
        
        # add paper counts using keyword matches only
        document_cluster_mapping['Paper_counts_matches'] = document_cluster_mapping\
            [document_cluster_mapping['Keyword_match'] == True]\
            .groupby(['Topic_label'])['Topic_label'].transform('count')
        
        # match each abstract with its title
        titles = dict(zip(tagged_db_entries['summary'].str.replace('\n', ' ',\
                            regex=True), tagged_db_entries['title']\
                                .str.replace('\n', '', regex=True).str.strip()))
        document_cluster_mapping['Title'] = \
            document_cluster_mapping['Document'].map(titles)
        
        # match each abstract with its authors
        authors = dict(zip(tagged_db_entries['summary'].str.replace('\n', ' ',\
                            regex=True), tagged_db_entries['authors']\
                                .str.replace(';', ',', regex=True)))
        document_cluster_mapping['Title'] = \
            document_cluster_mapping['Document'].map(authors)

        # match each abstract with a full PDF link
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
            break
        else:
            document_cluster_mapping_total = pd.concat([document_cluster_mapping_total, \
                            document_cluster_mapping]).reset_index(drop=True)

    # output table for debugging
    if manual_mode:
        save_results_as_xlsx(document_cluster_mapping_total, run_start_str)

    # save all topic modelling results to a JSON file
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
    parser.add_argument('-m', '--manual_mode', type=bool, \
            default=False, help='If set to True, additional debugging code is run.')

    main(parser.parse_args())

