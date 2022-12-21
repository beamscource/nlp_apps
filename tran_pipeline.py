
import argparse
import os
import re

from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi

# get a transcript from YT video
def transcribe(video_url, speech_rec_model):

    video_id = video_url.split("=")[1]
    video_transcript = YouTubeTranscriptApi.get_transcript(video_id)

    transcript_text = ''
    for i in video_transcript:
        transcript_text += ' ' + i['text']
    breakpoint()
    return transcript_text
# summarize and translate

# speech synthesis
def main(args):
    
    video_url = args.youtube_url
    speech_rec_model = args.speech_recognition_model
    summarize = args.summarize
    translate = args.translate
    trans_model = args.translation_model
    summ_model = args.summarization_model

    transcribe(video_url, speech_rec_model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transcribe YT videos und \
    summarize.')
    parser.add_argument('-yt', '--youtube_url', type=str, required=True, \
        help='URL for the YT video.')
    parser.add_argument('-sr', '--speech_recognition_model', type=str, \
        default='facebook/wav2vec2-base-960h', \
        help='HuggingFace model used for speech recognition.')
    parser.add_argument('-s', '--summarize', type=bool, default=True, \
        help='Transcript is summarized when set to True.')
    parser.add_argument('-t', '--translate', type=bool, default=True, \
        help='Summary is translated when set to True.')
    parser.add_argument('-sm', '--summarization_model', type=str, \
        default='google/pegasus-xsum', \
        help='HuggingFace model used for summarization.')
    parser.add_argument('-tm', '--translation_model', type=str, \
        default='Helsinki-NLP/opus-mt-en-de', \
        help='HuggingFace model used for translation.')
    main(parser.parse_args())