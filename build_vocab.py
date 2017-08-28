"""
Build vocabulary file
"""

import os
import json
import argparse
import h5py
import numpy as np
import string
from collections import Counter

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

__UNK_TOKEN = '<unk>'
__BOS_TOKEN = '<start>'
__EOS_TOKEN = '<end>'

def build_vocab(videos, word_count_threshold):

    # count up the number of words
    counter = Counter()
    for v in videos:
        for tokens in v['processed_tokens']:
            counter.update(tokens)
    
    cw = sorted([(count, w) for w, count in counter.iteritems()], reverse=True)
    logger.info('Top words and their counts: \n %s', '\n'.join(map(str, cw[:20])))

    unknown_words = [w for w, n in counter.iteritems() if n < word_count_threshold]
    unknown_count = sum(counter[w] for w in unknown_words)
    total_count = sum(counter.itervalues())

    # this is to make sure, word index (including unk) is greater than 0
    # because in the neuraltalk project the bos and eos are both NULL (0) tokens
    vocab = [__BOS_TOKEN, __EOS_TOKEN, __UNK_TOKEN]
    vocab.extend([w for w, n in counter.iteritems() if n >= word_count_threshold])
    
    logger.info('Total words: %d', total_count)
    logger.info('>> Number of unknown words: %d/%d = %.2f%%', len(unknown_words), len(counter), len(unknown_words) * 100.0 / len(counter))
    logger.info('>> Number of words in vocab (including <unk>): %d', len(vocab))
    logger.info('>> Number of UNKs: %d/%d = %.2f%%', unknown_count, total_count, unknown_count * 100.0 / total_count)
    
    return vocab

def main(input_json, output_json, word_count_threshold):

    videos = json.load(open(input_json, 'r'))

    logger.info('Creating the vocab')
    vocab = build_vocab(videos, word_count_threshold)
                       
    logger.info('Writing to %s', output_json)
    json.dump(vocab, open(output_json, 'w'))
    
######################################################################    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    parser.add_argument('input_json', type=str,
                        help='%_proprocessedtokens.json')
    parser.add_argument(
        'output_json', default='_vocab.json', help='output vocab file')
    
    parser.add_argument('--word_count_threshold', default=0, type=int,
                        help='only words that occur no less than this number of times will be put in vocab')

    args = parser.parse_args()
    logger.info('Input parameters: %s', args)
    
    start = datetime.now()
    main(args.input_json, args.output_json, args.word_count_threshold)
    
    logger.info('Time: %s', datetime.now() - start)
