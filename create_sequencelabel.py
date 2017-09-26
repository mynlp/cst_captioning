"""
Read caption from a json file 
Save as h5 format
"""

import os
import json
import argparse
import h5py
import numpy as np
import string
from random import shuffle, seed

import logging
from datetime import datetime
from build_vocab import __PAD_TOKEN, __UNK_TOKEN, __BOS_TOKEN, __EOS_TOKEN

logger = logging.getLogger(__name__)

def encode_captions(videos, max_length, wtoi):
    """ 
    encode all captions into one large array
    """

    N = len(videos)
    M = sum(len(v['final_captions']) for v in videos) # total number of captions

    label_arrays = []
    # note: these will be one-indexed
    label_start_ix = np.zeros(N, dtype=int)
    label_end_ix = np.zeros(N, dtype=int)
    label_length = np.zeros(M, dtype=int)
    label_to_video = np.zeros(M, dtype=int)
    counter = 0
    for i, v in enumerate(videos):
        n = len(v['final_captions'])
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype=int) # 0 is __PAD_TOKEN, implicitly
        for j, s in enumerate(v['final_captions']):
            
            label_length[counter+j] = min(max_length, len(s))
            label_to_video[counter+j] = i
            
            # truncated at max_length, thus the last token might be not the <eos>. 
            # any problem with this? 
            for k, w in enumerate(s):
                if k < max_length:
                    Li[j, k] = wtoi[w]

        label_arrays.append(Li)
        label_start_ix[i] = counter
        label_end_ix[i] = counter + n

        counter += n

    L = np.concatenate(label_arrays, axis=0)  # put all the labels together
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    # assert np.all(label_length > 0), 'error: some caption had no words?'

    logger.info('encoded captions to array of size %s', `L.shape`)
    return L, label_start_ix, label_end_ix, label_length, label_to_video

def main(vocab_json, captions_json, output_h5, max_length):

    # create the vocab
    vocab = json.load(open(vocab_json))
    
    # inverse table
    wtoi = {w: i for i, w in enumerate(vocab)} 
    
    videos = json.load(open(captions_json))

    logger.info('Select tokens in the vocab only')
    for v in videos:
        v['final_captions'] = []
        for txt in v['processed_tokens']:
            caption = [__BOS_TOKEN]
            caption += [w if w in wtoi else __UNK_TOKEN for w in txt]
            caption += [__EOS_TOKEN]
            v['final_captions'].append(caption)
    
    with h5py.File(output_h5, 'w') as of:
        if len(videos[0]['captions']) > 0:        
            logger.info('Encoding captions...')
            L, label_start_ix, label_end_ix, label_length, label_to_video = encode_captions(
                videos, max_length, wtoi)
            
            of.create_dataset('labels', dtype=int, data=L)
            of.create_dataset('label_start_ix', dtype=int, data=label_start_ix)
            of.create_dataset('label_end_ix', dtype=int, data=label_end_ix)
            of.create_dataset('label_length', dtype=int, data=label_length)
            of.create_dataset('label_to_video', dtype=int, data=label_to_video)
        else:
            logger.info('Caption not found! Skipped encoding captions.')
                    
        video_ids = [v['video_id'] for v in videos]        
        of['videos'] = np.array(video_ids, dtype=np.string_)
        of['vocab'] = np.array(vocab, dtype=np.string_)
        
        logger.info('Wrote to %s', output_h5)

######################################################################
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    parser.add_argument('vocab_json', default='_vocab.json',
                        help='vocab json file')
    parser.add_argument('captions_json', default='_proprocessedtokens',
                        help='_proprocessedtokens json file')
    parser.add_argument('output_h5', default='_sequencelabel.h5', help='output h5 file')
    
    parser.add_argument('--max_length', default=30, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')

    args = parser.parse_args()
    logger.info('Input parameters: %s', args)
    
    start = datetime.now()
    
    main(args.vocab_json, args.captions_json, args.output_h5, args.max_length)
    
    logger.info('Time: %s', datetime.now() - start)
