"""
Provide frame info
"""

from __future__ import print_function
import sys
import os

import json
import argparse
import logging
from datetime import datetime

import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s:%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'metadata',
        type=str,
        default='datainfo',
        help='meta file')
    parser.add_argument(
        'output',
        type=str,
        default='output/metadata/v2t2017_frameinfos.json',
        help='output file')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='input',
        help='root input dir')
    parser.add_argument(
        '--img_type',
        type=str,
        default='rgb',
        choices=[
            'rgb',
            'flow_x',
            'flow_y'],
        help='img_type')
    parser.add_argument(
        '--dataset',
        type=str,
        default='msrvtt2017',
        help='dataset')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=[
            'train',
            'val',
            'test'],
        help='img_type')
    parser.add_argument(
        '--sampling_step',
        type=int,
        default=8,
        help='uniform sample, need to be equal to the steps in the extracting feature code')

    args = parser.parse_args()
    logger.info('Command-line arguments: %s', args)

    start = datetime.now()

    logger.info('Loading metadata: %s', args.metadata)
    infos = json.load(open(args.metadata))
    videos = infos['videos']

    rel_frame_path = os.path.join(args.input_dir, args.dataset, 'flows')

    if args.dataset in ['tvvtt']:
        rel_frame_path = os.path.join(rel_frame_path, args.split)

    out = {}
    count_frame = 0
    for ii, video in enumerate(videos):
        video_id = video['video_id']

        logger.info('[%d/%d] Processing video: %s', ii, len(videos), video_id)

        frame_dir = os.path.join(rel_frame_path, video_id, args.img_type)
        frames = []
        for f in os.listdir(frame_dir):
            if f.endswith(".jpg"):
                frames.append(f)
        out[video_id] = frames[::args.sampling_step]
        if len(frames) == 0:
            logger.warning('Video has no frame: %s', video_id)
        count_frame += len(frames)

    logger.info('Total videos: %d', len(out))
    logger.info('Total frames: %d', count_frame)

    json.dump(out, open(args.output, 'w'))
    logger.info('Wrote to: %s', args.output)

    logger.info('Time: %s', datetime.now() - start)
