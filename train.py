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
import uuid
import logging
from datetime import datetime

from dataloader import DataLoader
from model import CaptionModel, CrossEntropyCriterion, RewardCriterion

import utils
import opts

import sys
sys.path.append("cider")
from pyciderevalcap.cider.cider import Cider
from pyciderevalcap.ciderD.ciderD import CiderD

logger = logging.getLogger(__name__)

def language_eval(predictions, cocofmt_file, opt):
    logger.info('>>> Language evaluating ...') 
    tmp_checkpoint_json = os.path.join(opt.model_file + str(uuid.uuid4()) + '.json')
    json.dump(predictions, open(tmp_checkpoint_json, 'w'))
    lang_stats = utils.language_eval(cocofmt_file, tmp_checkpoint_json)
    os.remove(tmp_checkpoint_json)
    return lang_stats
                
def train(model, criterion, optimizer, train_loader, val_loader, opt, rl_criterion=None):
    
    infos = {'iter': 0, 
             'epoch': 0, 
             'best_score': 0,
             'best_iter': 0,
             'best_epoch': opt.max_epochs
            }
    
    if os.path.isfile(opt.start_from) or os.path.isdir(opt.start_from):
        if os.path.isdir(opt.start_from):
            start_from_file = os.path.join(opt.start_from, os.path.basename(opt.model_file))
        else:
            start_from_file = opt.start_from
        logger.info('Loading state from: %s', start_from_file)
        checkpoint = torch.load(start_from_file)
        model.load_state_dict(checkpoint['model'])
        infos = checkpoint['infos']
        train_loader.set_current_epoch(infos['epoch'])
    
    checkpoint_checked = False
    scst_training = False
    
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
        
        if opt.use_ss == 1 and infos['epoch'] >= opt.use_ss_after:
            annealing_prob = opt.ss_k/(opt.ss_k + np.exp((infos['epoch'] - opt.use_ss_after)/opt.ss_k))
            opt.ss_prob = min(1 - annealing_prob, opt.ss_max_prob)
            model.set_ss_prob(opt.ss_prob)
                
        if opt.use_scst == 1 and infos['epoch'] >= opt.use_scst_after and not scst_training:
            logger.info('Start training using SCST objective...')
            scst_training = True
            CiderD_scorer = CiderD(df=opt.train_cached_tokens)
            # CiderD_scorer = Cider(df=opt.train_cached_tokens)
            # logger.info('loading gt refs: %s', train_loader.cocofmt_file)
            # gt_refs = utils.load_gt_refs(train_loader.cocofmt_file)
            
        optimizer.zero_grad()
        model.set_seq_per_img(train_loader.get_seq_per_img())
        
        if scst_training:
            # sampling from model distribution
            model_res, sample_logprobs = model.sample(feats, {'sample_max':0, 
                                                              'expand_feat': opt.expand_feat,
                                                              'temperature': 1.0})
            # greedy decoding baseline
            greedy_res, _ = model.sample([Variable(f.data, volatile=True) for f in feats],
                                         {'sample_max': 1, 'expand_feat': opt.expand_feat})
            
            model_sents = utils.decode_sequence(opt.vocab, model_res)
            greedy_sents = utils.decode_sequence(opt.vocab, greedy_res)

            for jj, sent in enumerate(zip(model_sents, greedy_sents)):
                if opt.expand_feat == 1:
                    video_id = data['ids'][jj//train_loader.get_seq_per_img()]
                else:
                    video_id = data['ids'][jj]
                logger.debug('[%d] video %s\n\t Model: %s \n\t Greedy: %s' %(jj, video_id, sent[0], sent[1]))
            
            reward, m_cider_score, g_cider_score = utils.get_self_critical_reward(model_res, greedy_res, data['gts'], CiderD_scorer)
            #reward, ciderd_score = utils.get_self_critical_reward(model_predictions, greedy_predictions, gt_refs, CiderD_scorer)
            loss = rl_criterion(model_res, sample_logprobs, Variable(torch.from_numpy(reward).float().cuda(), requires_grad=False))
            
        else:
            pred = model(feats, labels)
            loss = criterion(pred, labels[:,1:], masks[:,1:])
        
        loss.backward()
        clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()
        
        if infos['iter'] % opt.print_log_interval == 0:
            elapsed_time = time.time() - t_start
            
            log_info = [('Epoch', infos['epoch']),
                        ('Iter', infos['iter']),
                        ('Loss', loss.data[0])]
                        
            if scst_training and opt.use_scst == 1:
                log_info += [('Reward', np.mean(reward[:,0])),
                             ('Cider-D (m)', m_cider_score),
                            ('Cider-D (g)', g_cider_score)]
            if opt.use_ss == 1:
                log_info += [('ss_prob', opt.ss_prob)]
            
            log_info += [('Time', elapsed_time)]
            logger.info('%s', '\t'.join(['{}: {}'.format(k, v) for (k, v) in log_info]))
        
        infos['iter'] += 1
        
        if infos['epoch'] < train_loader.get_current_epoch():
            infos['epoch'] = train_loader.get_current_epoch()
            checkpoint_checked = False
            learning_rate = utils.adjust_learning_rate(opt, optimizer, infos['epoch'])
            logger.info('===> Learning rate: %f: ', learning_rate)

            
        if (infos['epoch'] >= opt.save_checkpoint_from and 
            infos['epoch'] % opt.save_checkpoint_every == 0 and 
            not checkpoint_checked):
            # evaluate the validation performance
            results = validate(model, criterion, val_loader, opt)
            logger.info('Validation output: %s', json.dumps(results['scores'], indent=4, sort_keys=True))
            infos.update(results['scores'])
            
            check_model(model, opt, infos)
            checkpoint_checked = True
        
        if (infos['epoch'] >= opt.max_epochs or \
            infos['epoch'] - infos['best_epoch'] > opt.max_patience):
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
        tmp_checkpoint_json = os.path.join(opt.model_file + str(uuid.uuid4()) + '.json')
        json.dump(predictions, open(tmp_checkpoint_json, 'w'))
        lang_stats = utils.language_eval(loader.cocofmt_file, tmp_checkpoint_json)
        os.remove(tmp_checkpoint_json)
    
    results['predictions'] = predictions
    results['scores']={'Loss': -loss}
    results['scores'].update(lang_stats)
    
    return results
    
def test(model, criterion, loader, opt):
    
    results = validate(model, criterion, loader, opt)
    logger.info('Test output: %s', json.dumps(results['scores'], indent=4))

    json.dump(results, open(opt.result_file, 'w'))
    logger.info('Wrote output caption to: %s ', opt.result_file)

def check_model(model, opt, infos):
    
    if opt.eval_metric == 'MSRVTT':
        current_score = infos['Bleu_4'] + infos['METEOR'] + infos['ROUGE_L'] + infos['CIDEr']
    else:
        current_score = infos[opt.eval_metric]
        
    # write the full model checkpoint as well if we did better than ever
    if current_score > infos.get('best_score', float('-inf')):
        infos['best_score'] = current_score
        infos['best_iter'] = infos['iter']
        infos['best_epoch'] = infos['epoch']

        logger.info('>>> Found new best [%s] score: %f, at iter: %d, epoch %d', 
                    opt.eval_metric, current_score, infos['iter'], infos['epoch'])
        
        torch.save({'model': model.state_dict(),
                    'infos': infos,
                    'opt': opt
                }, opt.model_file)
        logger.info('Wrote checkpoint to: %s', opt.model_file)
        
    else:
        logger.info('>>> Current best [%s] score: %f, at iter %d, epoch %d', 
                    opt.eval_metric, infos['best_score'], 
                    infos['best_iter'], 
                    infos['best_epoch'])
        
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
    
    xe_criterion = CrossEntropyCriterion()
    rl_criterion = RewardCriterion()
    
    if torch.cuda.is_available():
        model.cuda()
        xe_criterion.cuda()
        rl_criterion.cuda()
    
    logger.info('Start training...')
    start = datetime.now()
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    infos = train(model, xe_criterion, optimizer, train_loader, val_loader, opt, rl_criterion=rl_criterion)
    logger.info('Best val %s score: %f. Best iter: %d. Best epoch: %d', opt.eval_metric, 
                    infos['best_score'],  infos['best_iter'], infos['best_epoch'])
    
    logger.info('Training time: %s', datetime.now() - start)
    
    if opt.result_file is not None:
        logger.info('Start testing...')
        start = datetime.now()
        
        logger.info('Loading model: %s', opt.model_file)
        checkpoint = torch.load(opt.model_file)
        model.load_state_dict(checkpoint['model'])
        
        test(model, xe_criterion, test_loader, opt)
        logger.info('Testing time: %s', datetime.now() - start)

    