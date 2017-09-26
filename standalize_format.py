"""
Convert MSRVTT format to standard JSON
"""

import os
import json
import argparse
import string
import itertools

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def standalize_yt2t(input_file):
    """
    Use data splits provided by the NAACL2015 paper
    Ref: 
    """
    logger.info('Reading file: %s', input_file)
    lines = [line.rstrip('\n') for line in open(input_file)]
    lines = [line.split('\t') for line in lines]
    
    logger.info('Building caption dictionary for each video key')
    video_ids = []
    capdict = {}
    for line in lines:
        video_id = line[0] 
        if video_id in capdict:
            capdict[video_id].append(line[1])
        else:
            capdict[video_id] = [line[1]]
            video_ids.append(video_id)
    
    # create the json blob
    videos = []
    captions = []
    counter = itertools.count()
    for video_id in video_ids:
        
        vid = int(video_id[3:])
        jvid = {}
        jvid['category'] = 'unknown'
        jvid['video_id'] = video_id
        jvid['id'] = vid
        jvid['start_time'] = -1
        jvid['end_time'] = -1
        jvid['url'] = ''
        videos.append(jvid)
        
        for caption in capdict[video_id]:
            jcap = {}
            jcap['id'] = next(counter)
            jcap['video_id'] = vid
            jcap['caption'] = unicode(caption, errors='ignore')
            captions.append(jcap)
        
    out = {}
    out['info'] = {}
    out['videos'] = videos
    out['captions'] = captions
    
    return out

def standalize_msrvtt(input_file, dataset='msrvtt2016', split='train', val2016_json=None):
    """
    Supports both msrvtt2016 and msrvtt2017
    There is no official train/val set in MSRVTT2017:
    -> train2017 = train2016 + test2016
    -> val2017 = val2016
    """
    info = json.load(open(args.input_file))

    if split == 'val':
        split = 'validate'

    out = {}
    out['info'] = info['info']

    if args.dataset == 'msrvtt2017' and split == 'train':
        # loading all training videos and removing those that are in the val2016 set
        logger.info('Loading val2016 info: %s', val2016_json)
        info2016 = json.load(open(val2016_json))
        val2016_videos = [v for v in info2016['videos'] if v['split'] == 'validate']

        val2016_video_dict = {v['video_id']:v['id'] for v in val2016_videos}
        out['videos'] = [v for v in info['videos'] if v['video_id'] not in val2016_video_dict]

    else:
        out['videos'] = [v for v in info['videos'] if v['split'] == split]
    
    tmp_dict = {v['video_id']:v['id'] for v in out['videos']}
    out['captions'] = [{'id': c['sen_id'], 'video_id': tmp_dict[c['video_id']], 'caption': c['caption']} 
                       for c in info['sentences'] if c['video_id'] in tmp_dict]
    
    return out

def standalize_tvvtt(input_file, split='train2016'):
    """
    Standalize TRECVID V2T task
    Read from a metadata file generated in the v2t2017 project
    Basically there is no split in the v2t dataset, 
    Just consider each provided set as an independent dataset
    """
    split_mapping = {
        'train': 'train2016',
        'val': 'test2016',
        'test': 'test2017'
    }
    
    split = split_mapping[split]
    logger.info('Loading file: %s, split: %s', input_file, split)
    info = json.load(open(input_file))[split]
    
    out = {}
    out['info'] = {}
    videos = []
    for v in info['videos']:
        jvid = {}
        jvid['category'] = 'unknown'
        jvid['video_id'] = str(v)
        jvid['id'] = v
        jvid['start_time'] = -1
        jvid['end_time'] = -1
        jvid['url'] = ''
        videos.append(jvid)
    out['videos'] = videos
    out['captions'] = info['captions']
    
    return out

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', type=str, help='')
    parser.add_argument('output_json', type=str, help='')
    parser.add_argument('--split', type=str, help='')
    parser.add_argument('--dataset', type=str, default='yt2t', choices=['yt2t', 'msrvtt2016', 'msrvtt2017', 'tvvtt'], help='Choose dataset')
    parser.add_argument('--val2016_json', type=str, help='use valset from msrvtt2016 contest')

    args = parser.parse_args()
    logger.info('Input arguments: %s', args)
    
    start = datetime.now()
    
    if args.dataset == 'msrvtt2016':
        out = standalize_msrvtt(args.input_file, dataset=args.dataset, split=args.split)
    elif args.dataset == 'msrvtt2017':
        out = standalize_msrvtt(args.input_file, dataset=args.dataset, split=args.split, val2016_json=args.val2016_json)
    elif args.dataset == 'yt2t':
        out = standalize_yt2t(args.input_file)
    elif args.dataset == 'tvvtt':
        out = standalize_tvvtt(args.input_file, split=args.split)
    else:
        raise ValueError('Unknow dataset: %s', args.dataset)
        
    with open(args.output_json, 'w') as of:
        json.dump(out, of)
        
    logger.info('Time: %s', datetime.now() - start)
