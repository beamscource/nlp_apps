
# general imports
import argparse
import os
import re

# for progress bars
from tqdm import tqdm

# for downloding PDFs
import requests
from bs4 import BeautifulSoup

# for summarization and translation
from haystack.nodes import PDFToTextConverter
from haystack.nodes import TransformersSummarizer
from haystack.nodes import TransformersTranslator

#from haystack.utils import clean_wiki_text, convert_files_to_docs

def get_file_list(document_folder):

    if os.path.isdir(document_folder):
        file_list = []

    for file in os.listdir(document_folder):
        if file.endswith('.pdf'):
            file_list.append(file)
    
    return file_list

def download_pdfs(url, pdf_folder, number_pdfs):

    # define the base URL
    https, rest = url.split('//')
    base_url, *_ = rest.split('/')
    # https:// ...
    base_url = ''.join([https, '//', base_url])
    
    # request URL and get response object
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # extract and clean hyperlinks for PDF files
    links = soup.find_all('a')
    pdf_links = [re.findall(re.compile('href="(.*?)"'), str(link)) \
        for link in links if ('pdf' in link.get('href', []))]
    
    # combine relative URLs with the base URL
    complete_pdf_links = []
    for link in pdf_links:
        complete_pdf_links.append(''.join([base_url, link[0], '.pdf']))

    print(f'Downloading {len(complete_pdf_links[:number_pdfs])} PDF files.')

    for pdf_link in tqdm(complete_pdf_links[:number_pdfs]):

        response = requests.get(pdf_link)
        pdf_object = io.BytesIO(response.content).read()
        
        with open(os.path.join(pdf_folder, pdf_link.split('/')[-1]), \
            'wb') as f:
            f.write(pdf_object)

def convert_from_pdf(file_list):

    # https://docs.haystack.deepset.ai/docs/file_converters
    documents = []


    for file in tqdm(file_list):

        converter = PDFToTextConverter(
            remove_numeric_tables=True,
            valid_languages=["en"]
        )
        doc = converter.convert(file_path=Path(file), meta=None)
        documents.append(doc)
    
    return documents

def summarize_docs(documents):
    
    # https://docs.haystack.deepset.ai/docs/summarizer
    # TO_DO: experiment with different models
    summarizer = TransformersSummarizer(model_name_or_path='Helsinki-NLP/opus-mt-en-de')
    summaries = []
    
    for document in tqdm(documents):
        summary.content = summarizer.predict(documents=[document])
        summaries.append(summary)
    
    return summaries

def translate_docs(documents):

    # https://docs.haystack.deepset.ai/docs/translator

    # TO_DO choose model
    translator = TransformersTranslator(model_name_or_path='google/pegasus-xsum')
    translations = []

    for document in tqdm(documents):
        translation = translator.translate(documents=[document], query=None)
        translations.append(translation)
    
    return translations


def main(args):

    url = args.download_url
    document_folder = args.document_folder

    if not os.path.isdir(document_folder):
        os.makedirs(document_folder)

    if len([file for file in os.listdir(document_folder) \
        if file.endswith('.pdf')]) == 0:
        download_pdfs(url, document_folder)

    file_list = get_file_list(document_folder)
    documents = convert_from_pdf(file_list)

    if summarize:
        documents = summarize_docs(documents)
    
    if translate:
        documents = translate_docs(documents)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Haystack pipeline downloading \
             PDF files, converting them to text and processing their contents. The \
             results are printed to a PDF file.')

    parser.add_argument('-url', '--download_url', type=str, \
            default='https://arxiv.org/list/cs.AI/recent', \
            help='URL where to download PDF files.')
    parser.add_argument('-f', '--document_folder', type=str, default='pdf_files', \
            help='Folder for saving the PDF files locally.')
    parser.add_argument('-s', '--summarize', type=bool, default=True, \
            help='Texts are summarized when set to True.')
    parser.add_argument('-t', '--translate', type=bool, default=True, \
            help='Texts are translated when set to True.')

    main(parser.parse_args())