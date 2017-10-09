from __future__ import print_function

import torch
import json
import h5py
import os
import numpy as np
import random
import time

import logging
from datetime import datetime
logger = logging.getLogger(__name__)

class DataLoader():
    
    """Class to load video features and captions"""
    def __init__(self, opt):
        
        self.iterator = 0
        self.epoch = 0
        
        self.batch_size = opt.get('batch_size', 128)
        self.seq_per_img = opt.get('seq_per_img', 1)
        self.word_embedding_size = opt.get('word_embedding_size', 512)
        self.num_chunks = opt.get('num_chunks', 1)
        self.mode = opt.get('mode', 'train')
        self.cocofmt_file = opt.get('cocofmt_file', None)
        
        # open the hdf5 info file
        logger.info('DataLoader loading h5 file: %s', opt['label_h5'])
        self.label_h5 = h5py.File(opt['label_h5'], 'r')
        
        self.vocab = [i for i in self.label_h5['vocab']]
        self.videos = [i for i in self.label_h5['videos']]
        
        self.ix_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.num_videos = len(self.videos)
        self.index = range(self.num_videos)

        #load the json file which contains additional information about the dataset
        feat_h5_files = opt['feat_h5']
        logger.info('DataLoader loading h5 files: %s', feat_h5_files)
        self.feat_h5 = []
        self.feat_dims = []
        for ii, feat_h5_file in enumerate(feat_h5_files):
            self.feat_h5.append(h5py.File(feat_h5_files[ii], 'r'))
            self.feat_dims.append(self.feat_h5[ii][self.videos[0]].shape[0])
            
        self.num_feats = len(feat_h5_files)

        # load in the sequence data
        if 'labels' in self.label_h5.keys():
            self.seq_length = self.label_h5['labels'].shape[1]
            logger.info('max sequence length in data is: %d', self.seq_length)

            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.label_h5['label_start_ix']
            self.label_end_ix = self.label_h5['label_end_ix']
            assert(self.label_start_ix.shape[0] == self.label_end_ix.shape[0])
            self.has_label = True
        else:
            self.has_label = False
            
        if self.mode == 'train':
            self.shuffle_videos()

    def __del__(self):
        for f in self.feat_h5:
            f.close()
        self.label_h5.close()
    
    def get_batch(self):
        
        video_batch = []
        for dim in self.feat_dims:
            feat = torch.FloatTensor(self.batch_size, self.num_chunks, dim).zero_()
            video_batch.append(feat)

        if self.has_label:
            label_batch = torch.LongTensor(self.batch_size * self.seq_per_img, self.seq_length).zero_()
            mask_batch = torch.FloatTensor(self.batch_size * self.seq_per_img, self.seq_length).zero_()
        
        videoids_batch = []
        gts = []

        for ii in range(self.batch_size):
            idx = self.index[self.iterator]
            video_id = int(self.videos[idx])
            videoids_batch.append(video_id)

            for jj in range(self.num_feats):
                video_batch[jj][ii] = torch.from_numpy(np.array(self.feat_h5[jj][str(video_id)]))
            
            if self.has_label:
                # fetch the sequence labels
                ix1 = self.label_start_ix[idx]
                ix2 = self.label_end_ix[idx]
                ncap = ix2 - ix1 # number of captions available for this image
                assert ncap > 0, 'No captions!!'

                seq = torch.LongTensor(self.seq_per_img, self.seq_length).zero_()
                seq_all = torch.from_numpy(np.array(self.label_h5['labels'][ix1:ix2]))

                if ncap <= self.seq_per_img:
                    seq[:ncap] = seq_all[:ncap]
                    for q in range(ncap, self.seq_per_img):
                        ixl = np.random.randint(ncap)
                        seq[q] = seq_all[ixl]
                else:
                    randpos = torch.randperm(ncap)
                    for q in range(self.seq_per_img):
                        ixl = randpos[q]
                        seq[q] = seq_all[ixl]
                    
                il = ii*self.seq_per_img
                label_batch[il:il+self.seq_per_img] = seq
                
                # Used for reward evaluation
                gts.append(self.label_h5['labels'][self.label_start_ix[idx]: self.label_end_ix[idx]])

            self.iterator += 1
            if self.iterator >= self.num_videos:
                logger.info('===> Finished loading epoch %d', self.epoch)
                self.iterator = 0
                self.epoch += 1
                if self.mode == 'train':
                    self.shuffle_videos()
        
        data = {}
        data['feats'] = video_batch
        data['ids'] = videoids_batch
        
        if self.has_label:
            # + 1 here to count the <eos> token, because the <eos> token is set to 0
            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 1, label_batch)))
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
                
            data['labels'] = label_batch
            data['masks'] = mask_batch
            data['gts'] = gts

        return data

    def reset(self):
        self.iterator = 0
    
    def get_current_index(self):
        return self.iterator

    def set_current_index(self, index):
        self.iterator = index
    
    def get_vocab(self):
        return self.ix_to_word
    
    def get_vocab_size(self):
        return len(self.vocab)
    
    def get_feat_dims(self):
        return self.feat_dims
    
    def get_feat_size(self):
        return sum(self.feat_dims)
    
    def get_num_feats(self):
        return self.num_feats
    
    def get_seq_length(self):
        return self.seq_length
    
    def get_seq_per_img(self):
        return self.seq_per_img
    
    def get_num_videos(self):
        return self.num_videos
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_current_epoch(self):
        return self.epoch
    
    def set_current_epoch(self, epoch):
        self.epoch = epoch
    
    def shuffle_videos(self):
        np.random.shuffle(self.index)
    
    def get_cocofmt_file(self):
        return self.cocofmt_file