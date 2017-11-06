"""
Computer CIDEr (CIDEr-D) scores of all splits
It is used to as the inputs of the compute_dataslice function
and as pre-computed cider scores at training time

"""

import os
import sys
import json
import argparse
import string
import itertools
import numpy as np
from collections import OrderedDict

#sys.path.append("cider")
#from pyciderevalcap.cider.cider import Cider
#from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
#from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap

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

    #parser.add_argument('pred_file', type=str, help='')
    parser.add_argument('cocofmt_file', type=str, help='')
    parser.add_argument('output_pkl', type=str, help='')
    
    args = parser.parse_args()
    logger.info('Input arguments: %s', args)

    start = datetime.now()

    #logger.info('Loading prediction: %s', args.pred_file)
    #preds = json.load(open(args.pred_file))['predictions']
    #preds = {p['image_id']: [p['caption']] for p in preds}
    #scorer = CiderD(df=args.cached_tokens_file)
    
    scorer = Cider()
    logger.info('loading gt refs: %s', args.cocofmt_file)
    gt_refs = utils.load_gt_refs(args.cocofmt_file)
    
    logger.info('Computing score...')
    #score, scores = computer_score(preds, gt_refs, scorer)
    videos = sorted(gt_refs.keys())
    
    gt_scores = np.zeros((len(gt_refs), 20))
    for i in range(20):
        logger.info('taking caption: %d', i)
        preds_i = {v:[gt_refs[v][i]] for v in videos}
        score_i, scores_i = computer_score(preds_i, gt_refs, scorer)
        gt_scores[:,i] = scores_i
        #import pdb;pdb.set_trace()
        logger.info('score: %f', score_i)
        
    cPickle.dump({'cider': gt_scores}, open(
        args.output_pkl, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    logger.info('Time: %s', datetime.now() - start)
