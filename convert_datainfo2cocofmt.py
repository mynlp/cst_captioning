""" 
Convert input json to coco format to use COCO eval toolkit
"""

from __future__ import print_function
import argparse
from datetime import datetime
import logging
import json
import sys
import os.path
import random

logger = logging.getLogger(__name__)

# remove non-accii characters
def remove_nonaccii(s):
     s = ''.join([i if ord(i) < 128 else '' for i in s])
     return s
                    
if __name__ == '__main__':
    start = datetime.now()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')

    argparser = argparse.ArgumentParser(description = "Prepare image input in Json format for neuraltalk extraction and visualization")
    argparser.add_argument("input_json", type=str, help="Standalized datainfo file")
    argparser.add_argument("output_json", type=str, help="Output json in COCO format")
    argparser.add_argument("--max_caption", type=int, help="Max number of caption per video; default: 0 (all captions)", default=0)
    
    args = argparser.parse_args()
    
    logger.info('Loading input file: %s', args.input_json)
    infos = json.load(open(args.input_json))
            
    logger.info('Converting json data...')    
    
    imgs = [{'id': v['id']} for v in infos['videos']]
    
    if args.max_caption <= 0:
        anns = [{'caption': remove_nonaccii(s['caption']), \
                 'image_id': s['video_id'], \
                 'id': s['id']} \
                for s in infos['captions']]
    else:
        logger.info('Create dictionary of video captions')
        org_dict = {}
        for s in infos['captions']:
            org_dict.setdefault(s['video_id'], []).append(s['id'])
        
        sample_dict = {}
        logger.info('Randomly sample maximum %d caption(s) per video', args.max_caption)
        for k,v in org_dict.iteritems():
            sample_dict[k] = random.sample(org_dict[k], args.max_caption)
            
        anns = [{'caption': remove_nonaccii(s['caption']), \
                 'image_id': s['video_id'], \
                 'id': s['id']} \
                for s in infos['captions'] if s['id'] in sample_dict[s['video_id']]]
    
    out = {'images': imgs, 'annotations': anns, 'type': 'captions', 'info': infos['info'], 'licenses': 'n/a'}    
    
    logger.info('Saving...')    
    with open(args.output_json, 'w') as f:
        json.dump(out , f)
        
    logger.info('done')
    logger.info('Time: %s', datetime.now() - start)
    