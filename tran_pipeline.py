
import argparse
import os
import re

from youtube_transcript_api import YouTubeTranscriptApi

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from espnet2.bin.tts_inference import Text2Speech

import simpleaudio as sa # sudo apt-get install libasound2-dev
import numpy as np

def get_transcript(video_url, source_language, target_language):
    # see https://github.com/jdepoix/youtube-transcript-api
    video_id = video_url.split("=")[1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_transcript([f'{source_language}'])
    #translated_transcript = transcript.translate(f'{target_language}')
    video_transcript = transcript.fetch()

    transcript_text = ''
    for time_step in video_transcript:
        transcript_text += ' ' + \
            time_step['text'].replace('\n',' ').replace(' ,',',').replace(' .','.')

    return transcript_text.strip()

def extractive_summarize(transcript_text):
    # TO DO
    return transcript_text[:512]

def summarize(transcript_text, summarization_model):

    tokenizer = AutoTokenizer.from_pretrained(summarization_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model)

    prompted_text = ''.join(["summarize: ", transcript_text])
    inputs = tokenizer(prompted_text, return_tensors="pt").input_ids
    outputs = model.generate(inputs, max_new_tokens=300, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def speak(summary_text, synthesis_model):

    model = Text2Speech.from_pretrained(synthesis_model)
    speech  = model(summary_text)['wav']
    audio_array = speech.view(-1).cpu().numpy().astype(np.int16)
    play_obj = sa.play_buffer(audio_array, 1, 2, model.fs)
    play_obj.wait_done()
    breakpoint()

def main(args):
    
    video_url = args.youtube_url
    source_language = args.source_language
    target_language = args.target_language
    summarization_model = args.summarization_model
    synthesis_model = args.synthesis_model

    transcript_text = get_transcript(video_url, source_language, target_language)
    transcript_text = extractive_summarize(transcript_text)
    summary_text = summarize(transcript_text, summarization_model)
    speech = speak(summary_text, synthesis_model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transcribe YT videos und \
    summarize.')
    parser.add_argument('-yt', '--youtube_url', type=str, required=True, \
        help='URL for the YT video.')
    parser.add_argument('-sl', '--source_language', type=str, default='en', \
        help='Source language the YT video.')
    parser.add_argument('-tl', '--target_language', type=str, default='de', \
        help='Target language of the translation.')
    parser.add_argument('-sm', '--summarization_model', type=str, \
        default='google/pegasus-xsum', \
        help='HuggingFace model used for summarization.')
    parser.add_argument('-sym', '--synthesis_model', type=str, \
        default='espnet/kan-bayashi_ljspeech_vits', \
        help='HuggingFace model used for speech synthesis.')
    main(parser.parse_args())