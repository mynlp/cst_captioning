import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import numpy as np
import os
import sys
import time
import math
import json

import logging
from datetime import datetime

from dataloader import DataLoader
from model import CaptionModel, LanguageModelCriterion, RewardCriterion

import utils
import opts

import sys
sys.path.append("cider")
from pyciderevalcap.ciderD.ciderD import CiderD

logger = logging.getLogger(__name__)

def train(model, criterion, optimizer, train_loader, val_loader, opt, rl_criterion=None):
    
    infos = {'iter': 0, 
             'epoch': 0, 
             'best_scores': {},
             'best_iters': {},
             'best_epochs': {}
            }
    
    checked = False
    scst_training = False
    
    if opt.use_scst == 1:
        CiderD_scorer = CiderD(df=opt.train_cached_tokens)
        
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
        
        if scst_training and opt.use_scst == 1:
            gen_result, sample_logprobs = model.sample(feats, {'sample_max':0})
            # greedy decoding baseline
            greedy_res, _ = model.sample([Variable(f.data, volatile=True) for f in feats], {'sample_max': 1})
            reward, cider_score = utils.get_self_critical_reward(greedy_res, gen_result, data['gts'], CiderD_scorer)
            loss = rl_criterion(gen_result, sample_logprobs, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))
            
        else:
            pred = model(feats, labels)
            loss = criterion(pred, labels[:,1:], masks[:,1:])
        
        loss.backward()
        clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()
        
        if infos['iter'] % opt.print_log_interval == 0:
            elapsed_time = time.time() - t_start
            if scst_training and opt.use_scst == 1:
                logger.info('Epoch %d, Iter %d, Loss %f, Reward %f, Cider-D %.4f, (%.3fs/iter)', 
                        infos['epoch'], infos['iter'], loss.data[0], 
                            np.mean(reward[:,0]), cider_score, elapsed_time)
            else:
                logger.info('Epoch %d, Iter %d: %f (%.3fs/iter)', 
                        infos['epoch'], infos['iter'], loss.data[0], elapsed_time)
        
        if (infos['epoch'] >= opt.save_checkpoint_from and 
            infos['epoch'] % opt.save_checkpoint_every == 0 and 
            not checked):
            # evaluate the validation performance
            results = validate(model, criterion, val_loader, opt)
            logger.info('Validation output: %s', json.dumps(results['scores'], indent=4, sort_keys=True))
            infos.update(results['scores'])
            
            check_model(model, opt, infos)
            checked = True
        
        infos['iter'] += 1
        
        if infos['epoch'] < train_loader.get_current_epoch():
            infos['epoch'] = train_loader.get_current_epoch()
            checked = False
        
        if opt.use_scst == 1 and infos['epoch'] >= opt.use_scst_after:
            #logger.info('Start training using SCST objective...')
            scst_training = True
        
        if infos['epoch'] - infos['best_epochs'].get(opt.eval_metrics[0], sys.maxint) > opt.max_patience or \
            infos['epoch'] >= opt.max_epochs or \
            infos['iter'] >= opt.max_iters:
            logger.info('>>> Terminating...')
            break
            
    return infos
            
def validate(model, criterion, loader, opt):
    model.eval()
    loader.reset()
    
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
        if loader.has_label:
            labels = Variable(data['labels'], volatile=True)
            masks = Variable(data['masks'], volatile=True)
        
        if ii==(num_iters-1) and last_batch_size > 0:
            feats = [f[:last_batch_size] for f in feats]
            if loader.has_label:
                labels = labels[:last_batch_size * seq_per_img] # labels shape is DxN
                masks = masks[:last_batch_size * seq_per_img]
                
        if torch.cuda.is_available():
            feats = [feat.cuda() for feat in feats]
            if loader.has_label:
                labels = labels.cuda()
                masks = masks.cuda()
        
        if loader.has_label:
            pred = model(feats, labels)
            loss = criterion(pred, labels[:,1:], masks[:,1:])
            loss_sum += loss.data[0]
        
        seq, _ = model.sample(feats, {'beam_size': opt.beam_size})
        sents = utils.decode_sequence(opt.vocab, seq)
        
        for jj, sent in enumerate(sents):
            entry = {'image_id': data['ids'][jj], 'caption': sent}
            predictions.append(entry)
            logger.debug('[%d] video %s: %s' %(jj, entry['image_id'], entry['caption']))
                     
    loss = round(loss_sum/num_iters, 3)
    results = {}
    lang_stats = {}
    
    if opt.language_eval == 1 and loader.has_label:
        logger.info('>>> Language evaluating ...') 
        tmp_checkpoint_json = os.path.join(opt.checkpoint_path, 
                opt.id + '_tmp_predictions.json')
        json.dump(predictions, open(tmp_checkpoint_json, 'w'))
        lang_stats = utils.language_eval(loader.cocofmt_file, tmp_checkpoint_json)
        os.remove(tmp_checkpoint_json)
    
    results['predictions'] = predictions
    results['scores']={'Loss': -loss}
    results['scores'].update(lang_stats)
    
    return results
    
def test(model, criterion, loader, opt, infos={}):
    
    for ii, eval_metric in enumerate(opt.eval_metrics):
        if opt.test_only == 1:
            checkpoint_pkl = os.path.join(opt.checkpoint_path, opt.id + '_' + eval_metric + '.pkl')
            logger.info('Loading checkpoint: %s', checkpoint_pkl)
            model.load_state_dict(torch.load(checkpoint_pkl))
        else:
            logger.info('Best val %s score: %f. Best iter: %d. Best epoch: %d', eval_metric, 
                        infos['best_scores'].get(eval_metric, 0), 
                        infos['best_iters'].get(eval_metric, 0),
                        infos['best_epochs'].get(eval_metric, 0))

        results = validate(model, criterion, loader, opt)
        logger.info('Test output: %s', json.dumps(results['scores'], indent=4))
    
        checkpoint_json = os.path.join(opt.checkpoint_path, 
                                    opt.id + '_' + eval_metric + '_test_predictions.json')
        
        json.dump(results, open(checkpoint_json, 'w'))
        logger.info('Wrote output caption to: %s ', checkpoint_json)

def check_model(model, opt, infos):
    
    for ii, eval_metric in enumerate(opt.eval_metrics):
        if eval_metric == 'MSRVTT':
            current_score = infos['Bleu_4'] + infos['METEOR'] + infos['ROUGE_L'] + infos['CIDEr']
        else:
            current_score = infos[eval_metric]
        
        # write the full model checkpoint as well if we did better than ever
        if current_score > infos['best_scores'].get(eval_metric, float('-inf')):
            infos['best_scores'][eval_metric] = current_score
            infos['best_iters'][eval_metric] = infos['iter']
            infos['best_epochs'][eval_metric] = infos['epoch']

            logger.info('>>> Found new best [%s] score: %f, at iter: %d, epoch %d', 
                        eval_metric, current_score, infos['iter'], infos['epoch'])
            checkpoint_pkl = os.path.join(opt.checkpoint_path, opt.id + '_' + eval_metric + '.pth')
            torch.save(model.state_dict(), checkpoint_pkl)               
            logger.info('Wrote checkpoint to: %s', checkpoint_pkl)
            
            checkpoint_json = os.path.join(opt.checkpoint_path, 
                                    opt.id + '_' + eval_metric + '_val_predictions.json')
            json.dump(infos, open(checkpoint_json, 'w'))
            logger.info('Wrote output captions to: %s', checkpoint_json)
        else:
            logger.info('>>> Current best [%s] score: %f, at iter %d, epoch %d', 
                    eval_metric, infos['best_scores'][eval_metric], 
                        infos['best_iters'][eval_metric], 
                        infos['best_epochs'][eval_metric])
        
if __name__ == '__main__':
    
    opt = opts.parse_opts()

    logging.basicConfig(level=getattr(logging, opt.loglevel.upper()),
                        format='%(asctime)s:%(levelname)s: %(message)s')
    
    logger.info('Input arguments: %s', json.dumps(vars(opt), sort_keys=True, indent=4))
    
    # Set the random seed manually for reproducibility.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        
    train_opt = {'label_h5': opt.train_label_h5, 
        'batch_size': opt.batch_size,
        'feat_h5': opt.train_feat_h5,
        'cocofmt_file': opt.train_cocofmt_file,
        'seq_per_img': opt.train_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'train'
    }
    
    val_opt = {'label_h5': opt.val_label_h5, 
        'batch_size': opt.test_batch_size,
        'feat_h5': opt.val_feat_h5,
        'cocofmt_file': opt.val_cocofmt_file,
        'seq_per_img': opt.test_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'test'
    }
    
    test_opt = {'label_h5': opt.test_label_h5, 
        'batch_size': opt.test_batch_size,
        'feat_h5': opt.test_feat_h5,
        'cocofmt_file': opt.test_cocofmt_file,
        'seq_per_img': opt.test_seq_per_img,
        'num_chunks': opt.num_chunks,
        'mode': 'test'
    }
                
    train_loader = DataLoader(train_opt)
    val_loader = DataLoader(val_opt)
    test_loader = DataLoader(test_opt)
    
    opt.vocab = train_loader.get_vocab()
    opt.vocab_size = train_loader.get_vocab_size()
    opt.seq_length = train_loader.get_seq_length()
    opt.feat_dims = train_loader.get_feat_dims()
    
    logger.info('Building model...')
    model = CaptionModel(opt)
    
    xe_criterion = LanguageModelCriterion()
    
    if opt.use_scst == 1:
        rl_criterion = RewardCriterion()
    
    if torch.cuda.is_available():
        model.cuda()
        xe_criterion.cuda()
        if opt.use_scst == 1:
            rl_criterion.cuda()
    
    if opt.test_only == 1:
        test(model, xe_criterion, test_loader, opt)
        
    else:    
        if not os.path.exists(opt.checkpoint_path):
            os.makedirs(opt.checkpoint_path)
            
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
        if opt.use_scst == 1:
            infos = train(model, xe_criterion, optimizer, train_loader, val_loader, opt, rl_criterion=rl_criterion)
        else:
            infos = train(model, xe_criterion, optimizer, train_loader, val_loader, opt)
        test(model, criterion, test_loader, opt, infos)
        
