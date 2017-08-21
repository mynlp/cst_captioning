import sys
import os
import json

sys.path.append('tools/coco-caption')
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

import pdb
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

def language_eval(gold_file, pred_file)
    coco = COCO(gold_file)
    cocoRes = coco.loadRes(pred_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = round(score, 3)

    return out
