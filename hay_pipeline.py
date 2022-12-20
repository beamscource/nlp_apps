
''' A script to extract paper abstracts about AI topics (either from PDFs or
    directly from the web page), summarizing and translating them using Haystack
    components (https://docs.haystack.deepset.ai/docs/nodes_overview). Additional
    components (e.g., EntityExtractor) are easy to integrate. 
    
    The results are saved then to a single PDF.'''

# general imports
import argparse
import io
import os
import re
import sys

# for progress bars
from tqdm import tqdm

# for downloading PDFs
import requests
from bs4 import BeautifulSoup

# for saving PDFs
from fpdf import FPDF
import textwrap

# for summarization and translation
from haystack.nodes import PDFToTextConverter
from haystack import Document
from haystack.nodes import TransformersSummarizer
from haystack.nodes import TransformersTranslator

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

def convert_from_pdf(file_list, pdf_folder, number_pdfs):

    def extract_clean_text(document, extract):

        # extract specific number of words
        if extract == 'simple':
            #abstract_pattern = re.compile(r'Abstract(^(?:\S+\s+\n?){1,150})', re.DOTALL)
            abstract_pattern = re.compile(r'Abstract(.{1500})', re.DOTALL)
            abstract = re.findall(abstract_pattern, document[0].content)
            try:
                abstract = re.sub(r'-\n', '', abstract[0])
                abstract = abstract.lstrip('.')
                abstract = re.sub(r'\n', ' ', abstract).strip()
            except:
                abstract = document[0].content[50:1500]
                abstract = re.sub(r'-\n', '', abstract)
                abstract = re.sub(r'\n', ' ', abstract).strip()
            return abstract

        # case: abstract with keywords
        abstract_key_pattern = re.compile(r'Abstract(.*?)Keywords', re.DOTALL)
        if re.findall(abstract_key_pattern, document[0].content):
            abstract = re.findall(abstract_key_pattern, document[0].content)
            abstract = re.sub(r'-\n', '', abstract[0])
            abstract = re.sub(r'\n', ' ', abstract).strip()
            
            # extract and clean keywords
            keywords_pattern = re.compile(r'Keywords: (.*?)\n[A-Z]', re.DOTALL)
            keywords = re.findall(keywords_pattern, document[0].content)
            keywords = re.sub(r'-\n', '', keywords[0])
            keywords = re.sub(r'\n', ' ', keywords).strip()
            
            # combine abstract and keywords
            document = ' '.join([abstract, 'Keywords: ', keywords])
        
            return document

    # https://docs.haystack.deepset.ai/docs/file_converters
    documents = []

    print(f'Converting {len(file_list)} PDF files to text documents.')

    converter = PDFToTextConverter(
            remove_numeric_tables=True,
            valid_languages=["en"]
        )

    for file in tqdm(file_list[:number_pdfs]):

        document = converter.convert(file_path=os.path.join(pdf_folder, \
            file), meta=None)
        document = extract_clean_text(document, 'simple')
        documents.append(Document(document))

    return documents

def scrape_web_site(url):

    # define the base URL
    https, rest = url.split('//')
    base_url, *_ = rest.split('/')
    # https:// ...
    base_url = ''.join([https, '//', base_url])
    
    # request URL and get response object
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # extract and clean hyperlinks for abstracts
    links = soup.find_all('a')
    abstract_links = [re.findall(re.compile('href="(.*?)"'), str(link)) \
        for link in links if ('abs' in link.get('href', []))]
    
    # combine relative URLs with the base URL
    complete_abstract_links = []
    for link in abstract_links:
        complete_abstract_links.append(''.join([base_url, link[0]]))
    
    titles = []
    documents = []
    for abstract_link in complete_abstract_links:
        response = requests.get(abstract_link)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.get_text().split('] ')[1]
        abstract = soup.find_all('blockquote')[0].text
        abstract = abstract.split('Abstract: ')[1]
        abstract = re.sub(r'-\n', '', abstract)
        abstract = re.sub(r'\n', ' ', abstract).strip()
        titles.append(title)
        documents.append(Document(abstract))
        # url to the pdf
        # publication date
        # category
        # keywords
        # authors
    return titles, documents

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
    translations = translator.translate(documents=documents, query=None)

    return translations

def write_to_pdf(titles, abstracts, summaries, translations, pdf_file):
    # see https://stackoverflow.com/questions/10112244/convert-plain-text-to-pdf-in-python

    # formatting definitions
    a4_width_mm = 210
    pt_to_mm = 0.35
    fontsize_pt = 10
    fontsize_mm = fontsize_pt * pt_to_mm
    margin_bottom_mm = 10
    character_width_mm = 7 * pt_to_mm
    width_text = a4_width_mm / character_width_mm
    
    # create a PDF object
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(True, margin=margin_bottom_mm)
    pdf.add_page()
    pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
    pdf.set_font('DejaVu', '', size=fontsize_pt)

    print(f'Saving summaries for {len(abstracts)} documents to PDF.')

    for title, abstract, summary, translation in zip(titles, abstracts, \
        summaries, translations):

        pdf.cell(0, fontsize_mm, txt = 'Title', ln = 1)
        try:
            title_lines = textwrap.wrap(title, width_text)
        except:
            title_lines = textwrap.wrap('empty', width_text)
        for wrap in title_lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)
        
        pdf.cell(0, fontsize_mm, txt = 'Abstract', ln = 1)
        try:
            abstract_lines = textwrap.wrap(abstract, width_text)
        except:
            abstract_lines = textwrap.wrap('empty', width_text)
        for wrap in abstract_lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)

        # entities will go here

        pdf.cell(0, fontsize_mm, txt = 'Summary', ln = 1)
        try:
            summary_lines = textwrap.wrap(summary, width_text)
        except:
            summary_lines = textwrap.wrap('empty', width_text)
        for wrap in summary_lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)
        
        pdf.cell(0, fontsize_mm, txt = 'Translation', ln = 1)
        try:
            translation_lines = textwrap.wrap(translation, width_text)
        except:
            translation_lines = textwrap.wrap('empty', width_text)
        for wrap in translation_lines:
            pdf.cell(0, fontsize_mm, wrap, ln=1)

    pdf.output(pdf_file)

def main(args):

    url = args.download_url
    abstract_source = args.abstract_source
    pdf_folder = args.pdf_folder
    number_pdfs = args.number_pdfs
    summarize = args.summarize
    translate = args.translate
    trans_model = args.translation_model
    summ_model = args.summarization_model

    BASE_DIR = os.getcwd()

    if not os.path.isdir(pdf_folder):
        os.makedirs(pdf_folder)

    if abstract_source == 'web':
        titles, documents = scrape_web_site(url)
    else:
        if len([file for file in os.listdir(pdf_folder) \
            if file.endswith('.pdf')]) == 0:
            download_pdfs(url, pdf_folder, number_pdfs)

        file_list = get_file_list(pdf_folder)
        titles = file_list
        documents = convert_from_pdf(file_list, os.path.join(BASE_DIR, \
            pdf_folder), number_pdfs)

    # save original abstracts
    abstracts = []
    for abstract in documents:
        abstracts.append(abstract.content)

    if summarize:
        documents = summarize_docs(documents, summ_model)
        # save summaries
        summaries = []
        for document in documents:
            summaries.append(document.meta["summary"])

    # TO DO NER extraction

    if translate:
        documents = translate_docs(documents, trans_model)
        # save translations
        translations = []
        for document in documents:
            translations.append(document.content)

    # path to the summary PDF
    pdf_file = os.path.join(BASE_DIR, pdf_folder, 'summary.pdf')
    write_to_pdf(titles, abstracts, summaries, translations, pdf_file)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A Haystack pipeline \
            downloading PDF files, converting them to text and processing \
            their contents. The results are printed to a PDF file.')

    parser.add_argument('-url', '--download_url', type=str, \
            default='https://arxiv.org/list/cs.AI/recent', \
            help='URL where to extract abstracts/download PDF files.')
    parser.add_argument('-as', '--abstract_source', type=str, default='web', \
            help='Source for getting abstracts.')
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