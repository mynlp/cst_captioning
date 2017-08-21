import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import os
import time
import math

import logging
from datetime import datetime

from dataloader import DataLoader
import opts
from model import CaptionModel, LanguageModelCriterion
import utils

logger = logging.getLogger(__name__)

import pdb

def test(model, criterion, loader, opt):
    loader.reset()
    model.eval()
    
    num_videos = loader.get_num_videos()
    batch_size = loader.get_batch_size()
    num_iters = int(math.ceil(num_videos *1.0 / batch_size))
    last_batch_size = num_videos % batch_size
    seq_per_img = loader.get_seq_per_img()
    model.set_seq_per_img(seq_per_img)
    
    loss_sum = 0
    logger.info('#num_iters: %d, batch_size: %d, seg_per_image: %d', num_iters, batch_size, seq_per_img)
    predictions = []
    for ii in range(num_iters):
        data = loader.get_batch()
        feats = [Variable(feat, volatile=True) for feat in data['feats']]
        labels = Variable(data['labels'], volatile=True)
        masks = Variable(data['masks'], volatile=True)
        
        if ii==(num_iters-1) and last_batch_size > 0:
            feats = [f[:last_batch_size] for f in feats]
            if loader.has_label:
                labels = labels[:last_batch_size * seq_per_img] # labels shape is DxN
                masks = masks[:last_batch_size * seq_per_img]
                
        if torch.cuda.is_available():
            feats = [feat.cuda() for feat in feats]
            labels = labels.cuda()
            masks = masks.cuda()
        
        #feats_ = [f.clone() for f in feats] 
        if loader.has_label:
            pred = model(feats, labels)
            loss = criterion(pred, labels[:,1:], masks[:,1:])
            loss_sum += loss.data[0]
        
        #seq, _ = model.sample(feats, {'beam_size': opt.beam_size})
        seq, _ = model.sample(feats)
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        
        for jj, sent in enumerate(sents):
            entry = {'image_id': data['ids'][jj], 'caption': sent}
            predictions.append(entry)
            logger.debug('[%d] video %s: %s' %(jj, entry['image_id'], entry['caption']))
                     
    loss = loss_sum/num_iters
    results = {}
    
    lang_stats = {}
    
    if opt['language_eval'] == 1 and loader.hasLabel:
        logger.info(' ==> language evaluating ...') 
        coco_pred_json = os.path.join(opt.checkpoint_path, 
                opt.id + '_coco_predictions.json')
        json.dump(predictions, open(coco_pred_json, 'w'))
        lang_stats = utils.language_eval(evalopt.gold_ann_file, coco_pred_json)
        os.remove(coco_pred_json)
    
    results['predictions'] = predictions
    results['loss'] = loss
    results['lang_stats'] = lang_stats
    
    return results
    

def test_checkpoint(checkpoint_file, checkpoint_testpred_json):
    pass
    
def train(model, opt):
    pass
    
def stopping():
    pass

def model_selection():
    pass
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    opt = opts.parse_opts()
    logger.info('Input arguments: %s', opt)
    
    train_opt = {'label_h5': opt.train_label_h5, 
        'batch_size': opt.batch_size,
        'feat_h5': opt.train_feat_h5,
        'seq_per_img': opt.train_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'train'
    }
    train_loader = DataLoader(train_opt)
    
    val_opt = {'label_h5': opt.val_label_h5, 
        'batch_size': opt.test_batch_size,
        'feat_h5': opt.val_feat_h5,
        'seq_per_img': opt.test_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'test'
    }
    val_loader = DataLoader(val_opt)
    
    test_opt = {'label_h5': opt.test_label_h5, 
        'batch_size': opt.test_batch_size,
        'feat_h5': opt.test_feat_h5,
        'seq_per_img': opt.test_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'test'
    }
    test_loader = DataLoader(test_opt)
    
    opt.vocab_size = train_loader.get_vocab_size()
    opt.seq_length = train_loader.get_seq_length()
    opt.feat_dims = train_loader.get_feat_dims()
    
    logger.info('Loading model...')
    model = CaptionModel(opt)
    model.cuda()
    
    criterion = LanguageModelCriterion()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    iter = 0
    epoch = 0
    checkpoint_checked = False
    val_loss_history = {}
    val_lang_stats_history = {}
    
    while True:
        t_start = time.time()
        model.train()
        data = train_loader.get_batch()
        feats = [Variable(feat, volatile=False) for feat in data['feats']]
        labels = Variable(data['labels'], volatile=False)
        masks = Variable(data['masks'], volatile=False)
        
        if torch.cuda.is_available():
            feats = [feat.cuda() for feat in feats]
            labels = labels.cuda()
            masks = masks.cuda()
            
        optimizer.zero_grad()
        model.set_seq_per_img(train_loader.get_seq_per_img())
        pred = model(feats, labels)
        loss = criterion(pred, labels[:,1:], masks[:,1:])
        loss.backward()
        optimizer.step()
        
        if iter % opt.print_log_interval == 0:
            elapsed_time = time.time() - t_start
            logger.info('Epoch %d, Iter %d: %f (%.3fs/iter)', epoch, iter, loss.data[0], elapsed_time)
        
        # save checkpoint once in a while (or on final iteration)
        if epoch >= opt.save_checkpoint_from and epoch % opt.save_checkpoint_every == 0 and not checkpoint_checked:

            # evaluate the validation performance
            results = test(model, criterion, val_loader, opt)
            val_loss = results['loss']

            logger.info('Validation loss: %f', val_loss)
            #logger.info('Caption perplexity: %f', results['cap_perp'])

            if results['lang_stats']:
                logger.info('Validation lang_stats: %s', results['lang_stats'])      
                val_lang_stats_history[iter] = results['lang_stats']
            
            #json.dump(checkpoint, open(checkpoint_traininfo_json, 'w'))
            #logger.info('Wrote json checkpoint to: %s', checkpoint_traininfo_json)

            #stats = results['lang_stats'] 
            #stats['Loss'] = -val_loss
            #model_selection(stats, results['predictions'])
            checkpoint_checked = True
        
        iter += 1
        
        if epoch < train_loader.get_current_epoch():
            epoch = train_loader.get_current_epoch()
            checkpoint_checked = False
