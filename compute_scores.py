"""
Computer all metrics (BMCR) of all splits
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
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import logging
from datetime import datetime

import utils 
logger = logging.getLogger(__name__)
from six.moves import cPickle

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    #parser.add_argument('pred_file', type=str, help='')
    parser.add_argument('cocofmt_file', type=str, help='')
    parser.add_argument('output_pkl', type=str, help='')
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
    scorers = [
            (Bleu(), "Bleu_4"),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
    ]
        
    logger.info('loading gt refs: %s', args.cocofmt_file)
    gt_refs = utils.load_gt_refs(args.cocofmt_file)
    
    logger.info('Computing score...')
    #score, scores = computer_score(preds, gt_refs, scorer)
    videos = sorted(gt_refs.keys())
    
    gt_scores = {}
    for scorer, method in scorers:
        gt_scores[method] = np.zeros((len(gt_refs), args.seq_per_img))

    for i in range(args.seq_per_img):
        logger.info('taking caption: %d', i)
        preds_i = {v:[gt_refs[v][i]] for v in videos}
        
        # removing the refs at i
        if args.remove_in_ref:
            gt_refs_i = {v: gt_refs[v][:i] + gt_refs[v][i+1:] for v in videos}
        else:
            gt_refs_i = gt_refs
        
        for scorer, method in scorers:
            score_i, scores_i = scorer.compute_score(gt_refs_i, preds_i)
            
            if method == 'Bleu_4':
                score_i = score_i[-1]
                scores_i = scores_i[-1]
            
            # happens for BLeu and METEOR
            if type(scores_i) == list:
                scores_i = np.array(scores_i)
                
            gt_scores[method][:,i] = scores_i
            logger.info('%s: %f', method, score_i)
        
    cPickle.dump(gt_scores, open(
        args.output_pkl, 'w'), protocol=cPickle.HIGHEST_PROTOCOL)
    
    logger.info('Time: %s', datetime.now() - start)
