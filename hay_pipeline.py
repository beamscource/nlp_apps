
# general imports
import argparse
import io
import os
import re

# for progress bars
from tqdm import tqdm

# for downloading PDFs
import requests
from bs4 import BeautifulSoup

# for summarization and translation
from haystack.nodes import PDFToTextConverter
from haystack import Document
from haystack.nodes import TransformersSummarizer
from haystack.nodes import TransformersTranslator

#from haystack.utils import clean_wiki_text, convert_files_to_docs

def get_file_list(pdf_folder):

    if os.path.isdir(pdf_folder):
        file_list = []

    for file in os.listdir(pdf_folder):
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

def convert_from_pdf(file_list, pdf_folder):

    # https://docs.haystack.deepset.ai/docs/file_converters
    documents = []

    print(f'Converting {len(file_list)} PDF files to text documents.')

    converter = PDFToTextConverter(
            remove_numeric_tables=True,
            valid_languages=["en"]
        )

    # TO DO: clean text/target abstract
    for file in tqdm(file_list):

        document = converter.convert(file_path=os.path.join(pdf_folder, \
            file), meta=None)
        #documents.extend(document)
        documents.append(Document(document[0].content))

    return documents

def summarize_docs(documents, summ_model):

    # https://docs.haystack.deepset.ai/docs/summarizer

    print(f'Summarizing {len(documents)} documents.')
    # TO DO find a model with longer input sequence
    summarizer = TransformersSummarizer(model_name_or_path=summ_model)
    summaries = summarizer.predict(documents=documents)

    return summaries

def translate_docs(documents, trans_model):

    # https://docs.haystack.deepset.ai/docs/translator

    print(f'Translating {len(documents)} documents.')

    translator = TransformersTranslator(model_name_or_path=trans_model)
    translation = translator.translate(documents=documents, query=None)
    breakpoint()
    return translations

def write_to_pdf(documents):
    pass

def main(args):

    url = args.download_url
    pdf_folder = args.pdf_folder
    number_pdfs = args.number_pdfs
    summarize = args.summarize
    translate = args.translate
    trans_model = args.translation_model
    summ_model = args.summarization_model

    BASE_PATH = os.getcwd()

    if not os.path.isdir(pdf_folder):
        os.makedirs(pdf_folder)

    if len([file for file in os.listdir(pdf_folder) \
        if file.endswith('.pdf')]) == 0:
        download_pdfs(url, pdf_folder, number_pdfs)

    file_list = get_file_list(pdf_folder)
    documents = convert_from_pdf(file_list, os.path.join(BASE_PATH, \
        pdf_folder))

    if summarize:
        documents = summarize_docs(documents, summ_model)
    
    if translate:
        documents = translate_docs(documents, trans_model)

    # TO_DO write to pdf
    write_to_pdf(documents)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Haystack pipeline \
            downloading PDF files, converting them to text and processing \
            their contents. The results are printed to a PDF file.')

    parser.add_argument('-url', '--download_url', type=str, \
            default='https://arxiv.org/list/cs.AI/recent', \
            help='URL where to download PDF files.')
    parser.add_argument('-pdf', '--pdf_folder', type=str, default='pdf_files', \
            help='Folder for saving the PDF files locally.')
    parser.add_argument('-n', '--number_pdfs', type=int, default=3, \
            help='Number of PDF files to download.')
    parser.add_argument('-s', '--summarize', type=bool, default=True, \
            help='Texts are summarized when set to True.')
    parser.add_argument('-t', '--translate', type=bool, default=True, \
            help='Texts are translated when set to True.')
    parser.add_argument('-sm', '--summarization_model', type=str, \
            default='google/pegasus-xsum', \
            help='HuggingFace model used for summarization.')
    parser.add_argument('-tm', '--translation_model', type=str, \
            default='Helsinki-NLP/opus-mt-en-de', \
            help='HuggingFace model used for translation.')

    main(parser.parse_args())