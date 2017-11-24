"""
Compute CIDEr of each predicted sentence
Add add avg. ground truth human score on each video

"""

import os
import sys
import json
import argparse
import string
import itertools
import numpy as np
from collections import OrderedDict

sys.path.append("cider")
#from pyciderevalcap.cider.cider import Cider
from pyciderevalcap.ciderD.ciderD import CiderD

sys.path.append('coco-caption')
#from pycocotools.coco import COCO
#from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import logging
from datetime import datetime

import utils 
logger = logging.getLogger(__name__)
from six.moves import cPickle


def compute_score(gt_refs, hypo, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    # hypo = [{'image_id': p['image_id'], 'caption':[p['caption']]}
    #        for p in hypo]

    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    # ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}
    
    #preds_dict = [{'image_id': p['image_id'], 'caption':[p['caption']]} for p in hypo]
    
    score, scores = scorer.compute_score(gt_refs, hypo)

    return score, scores

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    #parser.add_argument('pred_file', type=str, help='')
    parser.add_argument('cocofmt_file', type=str, help='')
    parser.add_argument('bcmrscores_pkl', type=str, help='')
    parser.add_argument('pred_file', type=str, help='')
    parser.add_argument('cached_tokens_file', type=str, help='')
    parser.add_argument('out_pred_file', type=str, help='')
    parser.add_argument('--seq_per_img', type=int, default=20, help='Number of caption per image/video')
    parser.add_argument('--remove_in_ref', default=False, action='store_true', 
                        help='Remove current caption in the ref set')
    
    args = parser.parse_args()
    logger.info('Input arguments: %s', args)

    start = datetime.now()

    #logger.info('Loading prediction: %s', args.pred_file)
    #preds = json.load(open(args.pred_file))['predictions']
    #preds = {p['image_id']: [p['caption']] for p in preds}
    #scorer = CiderD(df=args.cached_tokens_file)
    
    logger.info('Setting up scorers...')
    #scorer = Cider()
    scorer = CiderD(df=args.cached_tokens_file)
        
    logger.info('loading gt refs: %s', args.cocofmt_file)
    gt_refs = utils.load_gt_refs(args.cocofmt_file)
    
    logger.info('loading bcmr scores: %s', args.bcmrscores_pkl)
    gt_scores = cPickle.load(open(args.bcmrscores_pkl))['CIDEr']
    std = np.std(gt_scores, 1)
    avg = np.mean(gt_scores, 1)
     
    logger.info('loading pred: %s', args.pred_file)    
    preds = json.load(open(args.pred_file))
    
    preds_dict = {v['image_id']: [v['caption']] for v in preds['predictions']}
    preds_ = [{'image_id': p['image_id'], 'caption':[p['caption']]} for p in preds['predictions']]
    
    gt_refs_ = {p['image_id']: gt_refs[p['image_id']] for p in preds['predictions']}
    
    logger.info('Computing overall score...')
    # this will compute the ngrams frequencies first, 
    # my bad but the deadline is close
    #score, scores = compute_score(gt_refs_, preds_, scorer)
    #logger.info('CIDEr: %f', score)
    
    videos = sorted(gt_refs.keys())

    logger.info('Computing score for each video...')
    for i,v in enumerate(videos):        
        preds_i = [{'image_id': v, 'caption':preds_dict[v]}]
        gt_refs_i = {v: gt_refs[v] } 
        score_i, scores_i = compute_score(gt_refs_i, preds_i, scorer)
        #import pdb; pdb.set_trace()
        assert(preds['predictions'][i]['image_id'] == v)
        preds['predictions'][i]['CIDEr'] = score_i
        preds['predictions'][i]['avgCIDEr'] = avg[i]
        preds['predictions'][i]['stdCIDEr'] = std[i]

    json.dump(preds, open(args.out_pred_file, 'w'))
    logger.info('Wrote to: %s...', args.out_pred_file)
    
    logger.info('Time: %s', datetime.now() - start)
