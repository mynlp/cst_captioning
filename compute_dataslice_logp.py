"""
Convert MSRVTT format to standard JSON
"""

import os
import sys
import json
import argparse
import string
import itertools
import numpy as np

sys.path.append("cider")
#from pyciderevalcap.cider.cider import Cider
#from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import logging
from datetime import datetime

import utils 
logger = logging.getLogger(__name__)
from six.moves import cPickle

def computer_score(hypo, gt_refs, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    #hypo = [{'image_id': p['image_id'], 'caption':[p['caption']]}
    #        for p in predictions]

    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    # ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}

    score, scores = scorer.compute_score(gt_refs, hypo)

    return score, scores

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file', type=str, help='')
    parser.add_argument('scores_pkl', type=str, help='')
    parser.add_argument('output_txt', default='out.txt', type=str, help='')
    
    args = parser.parse_args()
    logger.info('Input arguments: %s', args)

    start = datetime.now()
    
    logger.info('Loading prediction: %s', args.pred_file)
    test_avglogp = json.load(open(args.pred_file))['scores']['avglogp']
    
    logger.info('loading gt scores: %s', args.scores_pkl)
    gt_scores = cPickle.load(open(args.scores_pkl))
    
    #import pdb;pdb.set_trace()
    
    sorted_gt_scores = np.sort(gt_scores, axis=1)
    
    scores = []
    logger.info('Computing data slicing...')
    for i in range(20):
        logger.info('taking caption: %d', i)
    
        #import pdb;pdb.set_trace()
        score_i = np.mean(sorted_gt_scores[:,i:])
        logger.info('score: %f', score_i)
        scores.append(score_i)
    
    with open(args.output_txt, 'w') as f:
        for i, v in enumerate(scores):
            #f.write('{}, {}\n'.format(i, v))
            f.write('{}\n'.format(round(np.exp(-v), 3)))
        #f.write('{}\n'.format(round(test_avglogp, 3)))    
        
    logger.info('Time: %s', datetime.now() - start)
