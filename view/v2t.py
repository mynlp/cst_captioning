import os
import json
import argparse
import h5py
import numpy as np
import itertools
import cPickle

from flask import Flask, render_template, current_app, redirect, url_for, request
from flask_paginate import Pagination

from config import V2T_MODEL_DIR, V2T_METADATA_DIR, V2T_FEAT_DIR, MSRVTT_CATEGORIES, DATASETS
from SortableTable import SortableTable, Item

app = Flask(__name__)
app.config.from_pyfile('app.cfg')


def get_pagination(**kwargs):
    kwargs.setdefault('record_name', 'records')
    return Pagination(
        css_framework=current_app.config.get(
            'CSS_FRAMEWORK', 'bootstrap3'), link_size=current_app.config.get(
            'LINK_SIZE', 'sm'), show_single_page=current_app.config.get(
                'SHOW_SINGLE_PAGE', False), **kwargs)


def load_runs(dataset, exp):
    exp_dir = os.path.join(V2T_MODEL_DIR, exp)
    pred_jsons = []
    for f in os.listdir(exp_dir):
        if os.path.isfile(os.path.join(exp_dir, f)) and f.endswith(
                '_test.json'):
            pred_jsons.append(f)
    # app.logger.debug(pred_jsons)
    runs = {}
    for pred_json in pred_jsons:
        run_id = pred_json[:pred_json.find('.')]
        preds = json.load(open(os.path.join(exp_dir, pred_json)))
        runs[run_id] = preds
    return runs


def load_golds(dataset, split):
    meta_file = '{}_{}_datainfo.json'.format(dataset, split)
    gold_json = os.path.join(V2T_METADATA_DIR, meta_file)
    golds = json.load(open(gold_json))
    return golds

def load_bcmrscores(dataset, split):
    score_file = '{}_{}_evalscores.pkl'.format(dataset, split)
    score_file = os.path.join(V2T_METADATA_DIR, score_file)
    bcmrscores = cPickle.load(open(score_file))
    return bcmrscores

@app.route('/', methods=['GET'])
def index():
    data = {}
    data['datasets'] = DATASETS
    if request.method == 'GET':
        # app.logger.debug(request.args)
        if request.args:
            data['dataset'] = request.args.get('dataset')
            data['exp'] = request.args.get('exp')
            data['category'] = request.args.get('category')
    # app.logger.debug(data)
    if 'dataset' in data:
        if data['exp'] is None:
            data['exps'] = []
            for f in os.listdir(V2T_MODEL_DIR):
                if os.path.isdir(
                    os.path.join(
                        V2T_MODEL_DIR,
                        f)) and data['dataset'] in f:
                    data['exps'].append(f)
            if data['dataset'] == 'msrvtt':
                data['categories'] = MSRVTT_CATEGORIES
        elif data['dataset'] == 'msrvtt':
            return redirect("/videos?dataset={}&exp={}&category={}".format(
                data['dataset'], data['exp'], data['category']))
        else:
            return redirect(
                "/videos?dataset={}&exp={}".format(data['dataset'], data['exp']))

    return render_template('v2t_home.html', data=data)


@app.route('/videos/', defaults={'page': 1})
@app.route('/videos', defaults={'page': 1})
@app.route('/videos/page/<int:page>/')
@app.route('/videos/page/<int:page>')
def videos(page):

    config = {}
    data = {}

    config['datasets'] = DATASETS
    config['categories'] = MSRVTT_CATEGORIES
    config['splits'] = ['train', 'val', 'test']
    config['per_page'] = current_app.config.get('PER_PAGE', 10)
    config['page'] = page

    if request.method == 'GET' and request.args:
        data['dataset'] = request.args.get('dataset')
        data['split'] = request.args.get('split')
        data['category'] = request.args.get('category')

    if 'dataset' in data:
        video_ext = 'mp4'
        if data['dataset'] == 'yt2t':
            video_ext = 'avi'  # not playing at the moment

        golds = load_golds(data['dataset'], data['split'])
        bcmrscores = load_bcmrscores(data['dataset'], data['split'])
        
        videos = golds['videos']

        if data['dataset'] == 'msrvtt' and data['category'] != 'all':
            videos = [
                i for i in videos if MSRVTT_CATEGORIES[
                    i['category'] +
                    1] == data['category']]

        num_videos = len(videos)
        start_idx = (page - 1) * config['per_page']
        end_idx = min(page * config['per_page'], num_videos)

        videos = videos[start_idx:end_idx]
        for k in bcmrscores:
            bcmrscores[k] = bcmrscores[k][start_idx:end_idx]
            
        dataset_dir = data['dataset']
        if dataset_dir == 'msrvtt5c':
            dataset_dir = 'msrvtt'
        if dataset_dir == 'yt2t5c':
            dataset_dir = 'yt2t'

        for ii, video in enumerate(videos):
            video['keyframe'] = '{}/{}/flows/{}/rgb/000001.jpg'.format(
                current_app.config.get('V2T_IN_DIR'), dataset_dir, video['video_id'])
            video['videopath'] = '{}/{}/videos/{}.{}'.format(current_app.config.get(
                'V2T_IN_DIR'), dataset_dir, video['video_id'], video_ext)
            video['category'] = data['category']
            video['captions'] = [v['caption']
                                 for v in golds['captions'] if v['video_id'] == video['id']]

            video['bcmrscores'] = [round(bcmrscores['CIDEr'][ii,jj],3) for jj in range(20)]
            
        data['videos'] = videos

        data['pagination'] = get_pagination(page=page,
                                            per_page=config['per_page'],
                                            total=num_videos,
                                            record_name='videos',
                                            format_total=True,
                                            format_number=True,
                                            )

    return render_template('v2t_videos.html',
                           data=data,
                           config=config,
                           active_url='video-url'
                           )


@app.route('/experiments/', defaults={'page': 1}, methods=['GET'])
@app.route('/experiments', defaults={'page': 1})
@app.route('/experiments/page/<int:page>/')
@app.route('/experiments/page/<int:page>')
def experiments(page):

    config = {}
    data = {}

    config['datasets'] = DATASETS
    config['categories'] = MSRVTT_CATEGORIES
    config['per_page'] = current_app.config.get('PER_PAGE', 10)
    config['page'] = page

    if request.method == 'GET' and request.args:
        data['dataset'] = request.args.get('dataset')
        data['exp'] = request.args.get('exp')
        data['category'] = request.args.get('category')

    if 'dataset' in data:
        if data['exp'] is None:
            config['exps'] = []
            for f in os.listdir(V2T_MODEL_DIR):
                if os.path.isdir(
                    os.path.join(
                        V2T_MODEL_DIR,
                        f)) and 'cvpr2018' in f:
                    config['exps'].append(f)
        else:
            video_ext = 'mp4'
            if data['dataset'] == 'yt2t':
                video_ext = 'avi'  # not playing at the moment

            runs = load_runs(data['dataset'], data['exp'])

            preds = {k: v['predictions'] for k, v in runs.iteritems()}

            golds = load_golds(data['dataset'], 'test')

            video_ids = {v['id']: v['video_id'] for v in golds['videos']}

            if data['dataset'] == 'msrvtt' and data['category'] != 'all':
                for k, v in preds.iteritems():
                    preds[k] = [
                        i for i in v if MSRVTT_CATEGORIES[
                            i['category'] +
                            1] == data['category']]
            
            for k in preds:
                preds[k].sort(key=lambda x:x['stdCIDEr'])
            
            videos = preds[preds.keys()[0]]
            num_videos = len(videos)
            start_idx = (page - 1) * config['per_page']
            end_idx = min(page * config['per_page'], num_videos)

            videos = videos[start_idx:end_idx]

            dataset_dir = data['dataset']
            if dataset_dir == 'msrvtt5c':
                dataset_dir = 'msrvtt'
            if dataset_dir == 'yt2t5c':
                dataset_dir = 'yt2t'

            for ii, video in enumerate(videos):
                # string id -- to get file name
                video_id = video_ids[video['image_id']]
                video['keyframe'] = '{}/{}/flows/{}/rgb/000001.jpg'.format(
                    current_app.config.get('V2T_IN_DIR'), dataset_dir, video_id)
                video['videopath'] = '{}/{}/videos/{}.{}'.format(
                    current_app.config.get('V2T_IN_DIR'), dataset_dir, video_id, video_ext)
                video['id'] = video['image_id']  # number id
                video['category'] = data['category']
                video['captions'] = {}
                video['pred_topics'] = {}
                for k, v in preds.iteritems():
                    video['captions'][k] = '{}--CIDEr:{:.3f}--avg:{:.3f}--std:{:.3f}'.format(v[start_idx + ii]['caption'], v[start_idx + ii]['CIDEr'], v[start_idx + ii]['avgCIDEr'], v[start_idx + ii]['stdCIDEr'])
                    if 'topics' in v[start_idx + ii]:
                        video['pred_topics'][k] = v[start_idx + ii]['topics']

                video['gold_captions'] = [g['caption']
                                          for g in golds['captions'] if g['video_id'] == video['id']]

            data['videos'] = videos
            if 'topics' in videos[0]:
                data['topics'] = range(1, len(videos[0]['topics']) + 1)

                wpt = current_app.config.get('WORD_PER_TOPIC', 5)
                topiclist = {k: v.get('topics', None)
                             for k, v in runs.iteritems()}

                data['topiclist'] = {}
                for k, v in topiclist.iteritems():
                    if v:
                        data['topiclist'][k] = []
                        for ii, topic_id in enumerate(data['topics']):
                            topic_words = ", ".join(
                                [item['word'] for item in v[ii][:wpt]])
                            data['topiclist'][k].append(topic_words)

            data['pagination'] = get_pagination(page=page,
                                                per_page=config['per_page'],
                                                total=num_videos,
                                                record_name='videos',
                                                format_total=True,
                                                format_number=True,
                                                )

    return render_template('v2t_experiments.html',
                           data=data,
                           config=config,
                           active_url='exp-url'
                           )


@app.route('/leaderboard/', methods=['GET'])
@app.route('/leaderboard')
def leaderboard():

    config = {}
    data = {}

    config['datasets'] = DATASETS
    config['per_page'] = current_app.config.get('PER_PAGE', 10)

    if request.method == 'GET' and request.args:
        data['dataset'] = request.args.get('dataset')
        data['exp'] = request.args.getlist('exp')
        data['sort'] = request.args.get('sort', 'Id')
        data['reverse'] = (request.args.get('direction', 'asc') == 'desc')
        data['prefix'] = request.args.get('prefix', '')

    if 'dataset' in data:

        if len(data['exp']) == 0:
            config['exps'] = []
            for f in os.listdir(V2T_MODEL_DIR):
                if os.path.isdir(
                    os.path.join(
                        V2T_MODEL_DIR,
                        f)) and 'cvpr2018' in f:
                    config['exps'].append(f)
            config['exps'] = sorted(config['exps'])
        else:
            items = []
            counter = itertools.count(1)

            for exp in data['exp']:
                runs = load_runs(data['dataset'], exp)
                
                results = {k: v['scores'] for k, v in runs.iteritems() if k.startswith(data['prefix'])}

                items_ = [Item(next(counter), exp, data['dataset'], k, v)
                          for k, v in results.iteritems()]

                items.extend(items_)

            sorted_items = sorted(
                items,
                key=lambda x: getattr(x, data['sort']),
                reverse=data['reverse'])

            table = SortableTable(sorted_items, data['exp'], data['dataset'],
                                  sort_by=data['sort'],
                                  sort_reverse=data['reverse'],
                                  border=True)

            #return table.__html__()
            #import pdb; pdb.set_trace()
            
            table.save_markdown()
            
            data['table'] = table

    return render_template('v2t_leaderboard.html',
                           data=data,
                           config=config,
                           active_url='board-url'
                           )


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description="Options for running a Flask app")

    # Set up the command-line options
    argparser.add_argument("-H", "--host", type=str,
                           help="Hostname of the Flask app ",
                           default="127.0.0.1")
    argparser.add_argument("-P", "--port", type=int,
                           help="Port for the Flask app ",
                           default=5000)

    # Two options useful for debugging purposes, but
    # a bit dangerous so not exposed in the help message.
    argparser.add_argument("-d", "--debug",
                           action="store_true", dest="debug",
                           help=argparse.SUPPRESS)
    argparser.add_argument("-p", "--profile",
                           action="store_true", dest="profile",
                           help=argparse.SUPPRESS)

    args = argparser.parse_args()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )
