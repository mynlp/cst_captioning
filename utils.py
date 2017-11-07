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
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if isinstance(score, list):
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
    # hypo = {p['image_id']: [p['caption']] for p in predictions}

    # use with Cider provided by https://github.com/ruotianluo/cider
    hypo = [{'image_id': p['image_id'], 'caption':[p['caption']]}
            for p in predictions]

    # standard package requires ref and hypo have same keys, i.e., ref.keys()
    # == hypo.keys()
    ref = {p['image_id']: gt_refs[p['image_id']] for p in predictions}

    score, scores = scorer.compute_score(ref, hypo)

    return score, scores

# Input: seq, N*D numpy array, with element 0 .. vocab_size. 0 is END token.


def decode_sequence(ix_to_word, seq):
    N, D = seq.size()
    out = []
    for i in range(N):
        txt = ''
        for j in range(D):
            ix = seq[i, j]
            if ix > 0:
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
        out[metric] = round(score, 5)

    # restore the previous stdout
    sys.stdout = temp
    return out


def array_to_str(arr, use_eos=False):
    out = ''
    for i in range(len(arr)):
        if not use_eos and arr[i] == 0:
            break
        
        # skip the <bos> token    
        if arr[i] == 1:
            continue
            
        out += str(arr[i]) + ' '
        
        # return if encouters the <eos> token
        # this will also guarantees that the first <eos> will be rewarded
        if arr[i] == 0:
            break
            
    return out.strip()


def get_self_critical_reward2(model_res, greedy_res, gt_refs, scorer):

    model_score, model_scores = computer_score(model_res, gt_refs, scorer)
    greedy_score, greedy_scores = computer_score(greedy_res, gt_refs, scorer)
    scores = model_scores - greedy_scores

    m_score = np.mean(model_scores)
    g_score = np.mean(greedy_scores)

    #rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return m_score, g_score


def get_self_critical_reward(
        model_res,
        greedy_res,
        data_gts,
        CiderD_scorer,
        expand_feat=0,
        seq_per_img=20):
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
        gts[i] = [array_to_str(data_gts[i][j])
                  for j in range(len(data_gts[i]))]
    
    #_, scores = Bleu(4).compute_score(gts, res)
    #scores = np.array(scores[3])

    res = [{'image_id': i, 'caption': res[i]} for i in range(2 * batch_size)]
    if expand_feat == 1:
        gts = {i: gts[(i % batch_size) // seq_per_img]
               for i in range(2 * batch_size)}
    else:
        gts = {i: gts[i % batch_size] for i in range(2 * batch_size)}

    score, scores = CiderD_scorer.compute_score(gts, res)

    m_score = np.mean(scores[:batch_size])
    g_score = np.mean(scores[batch_size:])

    scores = scores[:batch_size] - scores[batch_size:]

    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)

    return rewards, m_score, g_score


def get_robust_critical_reward(
        model_res,
        data_gts,
        CiderD_scorer,
        scores=None,
        expand_feat=0,
        seq_per_img=20,
        num_robust=0,
        use_robust_baseline=1):
    
    """
    Args:
        num_robust: number of sentences to be removed
        use_baseline: use removed captions as baseline or not
        
    """
    if scores is None:
        batch_size = model_res.size(0)

        res = OrderedDict()
        model_res = model_res.cpu().numpy()
        for i in range(batch_size):
            res[i] = [array_to_str(model_res[i])]

        gts = OrderedDict()
        for i in range(len(data_gts)):
            gts[i] = [array_to_str(data_gts[i][j])
                      for j in range(len(data_gts[i]))]

        res = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]
        if expand_feat == 1:
            gts = {i: gts[(i % batch_size) // seq_per_img]
                   for i in range(batch_size)}
        else:
            gts = {i: gts[i % batch_size] for i in range(batch_size)}

        _, scores = CiderD_scorer.compute_score(gts, res)
        
        scores = scores.reshape(-1, seq_per_img)
        
    if num_robust > 0:
        # use removed sentences as baseline
        
        sorted_scores = np.sort(scores, axis=1)
        
        sorted_idx = np.argsort(scores, axis=1)
        
        m_score = np.mean(sorted_scores[:,num_robust:])
        b_score = np.mean(sorted_scores[:,:num_robust])
        
        for ii in range(scores.shape[0]):
            if use_robust_baseline == 1:
                # note: may need to reculate b, use the max value, 
                # rathan the average, as in the else statement
                b = np.mean(sorted_scores[ii,:num_robust])
                scores[ii] = scores[ii] - b
            else:
                # to turn off backprobs
                # however, negative scores may be also be helpful to learn
                b = sorted_scores[ii, num_robust]
                #scores[ii] = max(scores[ii] - b, 0)
                scores[ii][scores[ii]<=b] = 0
                
    else:
        m_score = np.mean(scores)
        b_score = 0
    
    scores = scores.reshape(-1)
    rewards = np.repeat(scores[:, np.newaxis], model_res.shape[1], 1)
    
    return rewards, m_score, b_score
