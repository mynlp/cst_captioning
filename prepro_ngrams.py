"""

"""

import os
import json
import argparse
from six.moves import cPickle
from collections import defaultdict

import logging
from datetime import datetime
from build_vocab import __PAD_TOKEN, __UNK_TOKEN, __BOS_TOKEN, __EOS_TOKEN

logger = logging.getLogger(__name__)


def precook(s, n=4, out=False):
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = s.split()
    counts = defaultdict(int)
    for k in xrange(1, n + 1):
        for i in xrange(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):  # lhuang: oracle will call with "average"
    '''Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    '''
    return [precook(ref, n) for ref in refs]


def create_crefs(refs):
    crefs = []
    for ref in refs:
        # ref is a list of 5 captions
        crefs.append(cook_refs(ref))
    return crefs


def compute_doc_freq(crefs):
    '''
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    '''
    document_frequency = defaultdict(float)
    for refs in crefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for (
                ngram, count) in ref.iteritems()]):
            document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def build_dict(videos, wtoi):

    count_videos = 0

    refs_words = []
    refs_idxs = []
    for v in videos:
        ref_words = []
        ref_idxs = []
        for sent in v['final_captions']:
            ref_words.append(' '.join(sent))
            ref_idxs.append(' '.join([str(wtoi[_]) for _ in sent]))
        refs_words.append(ref_words)
        refs_idxs.append(ref_idxs)
        count_videos += 1
        
    logger.info('total videos: %d', count_videos)

    ngram_words = compute_doc_freq(create_crefs(refs_words))
    ngram_idxs = compute_doc_freq(create_crefs(refs_idxs))
    return ngram_words, ngram_idxs, count_videos


def main(vocab_json, captions_json, output_pkl, save_words=False):

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

    ngram_words, ngram_idxs, ref_len = build_dict(videos, wtoi)

    logger.info('Saving index to: %s', output_pkl)
    cPickle.dump({'document_frequency': ngram_idxs, 'ref_len': ref_len}, open(
        output_pkl, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    if save_words:
        output_file = output_pkl.replace('.pkl', '_words.pkl', 1)
        logger.info('Saving word to: %s', output_file)
        cPickle.dump({'document_frequency': ngram_words, 'ref_len': ref_len}, open(
            output_file, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('vocab_json', default='_vocab.json',
                        help='vocab json file')
    parser.add_argument('captions_json', default='_proprocessedtokens',
                        help='_proprocessedtokens json file')
    parser.add_argument(
        'output_pkl',
        default='_pkl',
        help='save idx frequencies')
    
    parser.add_argument(
        '--output_words',
        action='store_true',
        help='optionally saving word frequencies')
    
    args = parser.parse_args()
    
    start = datetime.now()
    
    main(args.vocab_json, args.captions_json, args.output_pkl, save_words=args.output_words)
    
    logger.info('Time: %s', datetime.now() - start)
