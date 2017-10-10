import sys
import os
import json

import numpy as np
from collections import OrderedDict

sys.path.append('coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

import cPickle

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every [lr_update] epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores

def load_gt_refs(cocofmt_file):
    d = json.load(open(cocofmt_file))
    out = {}
    for i in d['annotations']:
        out.setdefault(i['image_id'], []).append(i['caption'])
    return out
        
def computer_score(predictions, gt_refs, scorer):
    # use with standard package https://github.com/tylin/coco-caption
    hypo = {p['image_id']: [p['caption']] for p in predictions}  
    
    # use with Cider provided by https://github.com/ruotianluo/cider
    # hypo = [{'image_id':p['image_id'], 'caption':[p['caption']]} for p in predictions]
    
    # standard package requires ref and hypo have same keys, i.e., ref.keys() == hypo.keys()
    ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}
    
    score, scores = Cider().compute_score(ref, hypo)
    
    import pdb; pdb.set_trace()
    
    return score, scores
    
# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.
def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i,j]
            if ix > 0 :
                if j >= 1:
                    txt = txt + ' '
                txt = txt + ix_to_word[ix]
            else:
                break
        out.append(txt)
    return out

def language_eval(gold_file, pred_file):
    
    # save the current stdout
    temp = sys.stdout 
    sys.stdout = open(os.devnull, 'w')

    coco = COCO(gold_file)
    cocoRes = coco.loadRes(pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = round(score, 3)

    # restore the previous stdout    
    sys.stdout = temp
    return out


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        if arr[i] == 0:
            break
        if arr[i] == 1:    
            continue
        out += str(arr[i]) + ' '
    return out.strip()

def get_self_critical_reward_(model_res, greedy_res, gt_refs, CiderD_scorer):
    
    model_score, model_scores = computer_score(model_res, gt_refs, CiderD_scorer)
    greedy_score, greedy_scores = computer_score(greedy_res, gt_refs, CiderD_scorer)
    
    scores = model_scores - greedy_scores

    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)
    
    return rewards, model_score


def get_self_critical_reward(model_res, greedy_res, data_gts, CiderD_scorer):
    batch_size = model_res.size(0)

    res = OrderedDict()
    
    model_res = model_res.cpu().numpy()
    greedy_res = greedy_res.cpu().numpy()
    for i in range(batch_size):
        res[i] = [array_to_str(model_res[i])]
    for i in range(batch_size):
        res[batch_size + i] = [array_to_str(greedy_res[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])
    res = [{'image_id':i, 'caption': res[i]} for i in range(2 * batch_size)]
    gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}
    
    _, scores = CiderD_scorer.compute_score(gts, res)
    
    m_score = np.mean(scores[:batch_size])
    g_score = np.mean(scores[batch_size:])
    
    scores = scores[:batch_size] - scores[batch_size:]
    
    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)
    
    return rewards, m_score, g_score
