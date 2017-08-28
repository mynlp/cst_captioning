"""
Preprocessing datainfo
tokenize, lowercase, etc.

"""

import os
import json
import argparse
import h5py
import numpy as np
import string
import nltk

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def prepro_captions(videos):

    logger.info("Preprocessing %d videos", len(videos))
    for i, v in enumerate(videos):
        v['processed_tokens'] = []
        if i % 100 == 0:
            logger.info("%d/%d video processed", i, len(videos))
            
        for caption in v['captions']:
            caption_ascii = ''.join(
                [ii if ord(ii) < 128 else '' for ii in caption])
            tokens = str(caption_ascii).lower().translate(
                None, string.punctuation).strip().split()
            v['processed_tokens'].append(tokens)

def main(input_json, output_json):

    infos = json.load(open(input_json, 'r'))
    annots = infos['captions']

    logger.info('group annotations by video')
    vtoa = {}
    for ann in annots:
        vtoa.setdefault(ann['video_id'], []).append(ann)

    logger.info('create the json blob')
    videos = []
    for i, v in enumerate(infos['videos']):
        
        jvid = {}
        jvid['category'] = v.get('category', 'unknown')
        jvid['video_id'] = v['id']

        sents = []
        annotsi = vtoa.get(v['id'], []) # at test time, there is no captions
        for a in annotsi:
            sents.append(a['caption'])
        jvid['captions'] = sents
        videos.append(jvid)

    logger.info('Tokenizing and preprocessing')
    prepro_captions(videos)
    
    logger.info('Writing to: %s', output_json)
    json.dump(videos, open(output_json, 'w'))

######################################################################    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    parser.add_argument('input_json', type=str,
                        help='standalized input json')
    parser.add_argument(
        'output_json', type=str, help='output tokenized json file')
    
    args = parser.parse_args()
    logger.info('Input arguments: %s', args)
    
    start = datetime.now()
    main(args.input_json, args.output_json)
    
    logger.info('Time: %s', datetime.now() - start)
