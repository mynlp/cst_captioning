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
    parser.add_argument('cocofmt_file', type=str, help='')
    parser.add_argument('scores_pkl', type=str, help='')
    parser.add_argument('output_txt', default='out.txt', type=str, help='')
    
    args = parser.parse_args()
    logger.info('Input arguments: %s', args)

    start = datetime.now()

    logger.info('Loading prediction: %s', args.pred_file)
    preds = json.load(open(args.pred_file))['predictions']
    preds = {p['image_id']: [p['caption']] for p in preds}
    
    scorer = Cider()
    logger.info('loading gt refs: %s', args.cocofmt_file)
    gt_refs = utils.load_gt_refs(args.cocofmt_file)
    
    logger.info('loading gt scores: %s', args.scores_pkl)
    gt_scores = cPickle.load(open(args.scores_pkl))['cider']
    sorted_idx = gt_scores.argsort(axis=1)
    
    videos = sorted(gt_refs.keys())
    
    scores = []
    logger.info('Computing data slicing...')
    for i in range(20):
        logger.info('taking caption: %d', i)
        gt_refs_i = {}
        for j,v in enumerate(videos):
            tmp_array = np.array(gt_refs[v])[sorted_idx[j][i:]]
            gt_refs_i[v] = tmp_array.tolist()
            
        #import pdb;pdb.set_trace()
        score_i, scores_i = computer_score(preds, gt_refs_i, scorer)
        logger.info('score: %f', score_i)
        scores.append(score_i)
    
    with open(args.output_txt, 'w') as f:
        for i, v in enumerate(scores):
            f.write('{}, {}\n'.format(i, v))
        
    logger.info('Time: %s', datetime.now() - start)
