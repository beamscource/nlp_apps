
import argparse
import os
import re

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

def get_german_transcript(video_url, language):

    video_id = video_url.split("=")[1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript([f'{language}'])
    translated_transcript = transcript.translate('de')
    video_transcript = translated_transcript.fetch()

    transcript_text = ''
    for i in video_transcript:
        transcript_text += ' ' + i['text'].replace('\n','').strip()
    breakpoint()
    return transcript_text

# summarize 

# speech synthesis
def main(args):
    
    video_url = args.youtube_url
    language = args.video_language
    summ_model = args.summarization_model

    get_german_transcript(video_url, language)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transcribe YT videos und \
    summarize.')
    parser.add_argument('-yt', '--youtube_url', type=str, required=True, \
        help='URL for the YT video.')
    parser.add_argument('-vl', '--video_language', type=str, default='en', \
        help='URL for the YT video.')
    parser.add_argument('-sm', '--summarization_model', type=str, \
        default='google/pegasus-xsum', \
        help='HuggingFace model used for summarization.')
    main(parser.parse_args())