""" 
TRECVID format:
video_id1 caption
video_id2 caption

...
video_idN caption

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


if __name__ == '__main__':
    start = datetime.now()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')

    argparser = argparse.ArgumentParser(description = "Prepare image input in Json format for neuraltalk extraction and visualization")
    argparser.add_argument("input_json", type=str, help="Standalized datainfo file")
    argparser.add_argument("output_txt", type=str, help="Output txt")
    
    args = argparser.parse_args()
    
    logger.info('Loading input file: %s', args.input_json)
    infos = json.load(open(args.input_json))
            
    logger.info('Converting json data...')    
    with open(args.output_txt, 'w') as f:
        for p in infos['predictions']:
            f.write('{} {}\n'.format(p['image_id'], p['caption']))
        
    logger.info('done')
    logger.info('Time: %s', datetime.now() - start)
    