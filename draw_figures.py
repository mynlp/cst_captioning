"""
Demo of a simple plot with a custom dashed line.

A Line object's ``set_dashes`` method allows you to specify dashes with
a series of on/off lengths (in points).
"""

from __future__ import print_function
import sys
import os, glob
import argparse
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import matplotlib.ticker as ticker


def draw_cider(input_dir):
    
    #methods = ['XE', 'CST_GT_None', 'CST_RL_Greedy', 'CST_RL_SCB']
    models = ['XE', 'CST_GT_None', 'SCST', 'CST_MS_Greedy', 'CST_MS_HCB', 'CST_MS_MCB']
    legend = ['XE', 'CST_GT_None', 'SCST', 'CST_MS_Greedy', 'CST_MS_SCB', 'CST_MS_SCB(*)']
    
    y_list = []
    for m in models:
        data_file = os.path.join(input_dir, m + '_dataslice.txt')
        lines = [float(line) for line in open(data_file)]
        y_list.append(lines)
    
    x = range(1, len(y_list[0])+1)
    x_labels = range(20,0,-1)
    
    fig = plt.figure( figsize=(6, 4), dpi=120)
    ax = fig.add_subplot(111)
    
    color_list = ['k', 'm', 'b', 'g', 'c', 'r', 'y']
    marker_list = ['>', 'd', 'h', 's', '*', 'v', '.']
    
    lines = [None]*len(y_list)
    for idx,y in enumerate(y_list):
        lines[idx] = ax.plot(x, y, '-', linewidth=2, marker=marker_list[idx], color=color_list[idx])
        #if idx == 1:
        #    ax.annotate(str(y[0]),xy=(x[0],y[0]),xytext=(-20,5), textcoords='offset points')
        #    ax.annotate(str(y[3]),xy=(x[3],y[3]),xytext=(-20,5), textcoords='offset points')
        

    plt.grid()
    
    #ax.set_xlim(0.5, len(x)+ 0.5)
    #ax.set_ylim(0.25, 0.8)
    #ax.set_ylabel('CIDEr')
    ax.yaxis.set_label_coords(-0.04, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.set_xlabel('Top CIDEr captions in reference set')
    
    ax.annotate(str(1),xy=(1,4))
    
    plt.xticks(x, x_labels)
    xtickNames = ax.get_xticklabels()
    plt.setp(xtickNames, rotation=0, fontsize=10)
    plt.ylabel('CIDEr', rotation=0)
    
    #ax.xaxis.set_label_coords(1.05, -0.025)
    ax.legend( (lines[0][0], lines[1][0], lines[2][0], lines[3][0], lines[4][0], lines[5][0]), \
        tuple(legend), ncol=2, fontsize=11,loc='upper left')
    
    #plt.savefig('exp1.png',bbox_inches='tight', pad_inches=0)

    plt.show()

    
def draw_logp(input_dir):
    legend = ['XE', 'CST_GT_None', 'CST_GT_HCB', 'CST_GT_R'] #, 'CST_MS_Greedy', 'CST_MS_HCB', 'CST_MS_MCB']
    
    y_list = []
    for m in legend:
        data_file = os.path.join(input_dir, m + '_dataslice_logp.txt')
        lines = [float(line) for line in open(data_file)]
        y_list.append(lines[:-1])
    
    x = range(1, len(y_list[0])+1)
    x_labels = range(20,0,-1)
    
    fig = plt.figure( figsize=(6, 4), dpi=80)
    ax = fig.add_subplot(111)
    
    color_list = ['b', 'g', 'c', 'm', 'r']
    marker_list = ['>', 'd', 'h', 's', '*']
    
    lines = [None]*len(y_list)
    for idx,y in enumerate(y_list):
        lines[idx] = ax.plot(x, y, '-', linewidth=2, marker=marker_list[idx], color=color_list[idx])
        #if idx == 1:
        #    ax.annotate(str(y[0]),xy=(x[0],y[0]),xytext=(-20,5), textcoords='offset points')
        #    ax.annotate(str(y[3]),xy=(x[3],y[3]),xytext=(-20,5), textcoords='offset points')
        

    plt.grid()
    
    #ax.set_xlim(0.5, len(x)+ 0.5)
    #ax.set_ylim(0.25, 0.8)
    #ax.set_ylabel('CIDEr')
    ax.yaxis.set_label_coords(-0.04, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.set_xlabel('Number of high consensus captions in the reference set')
    
    ax.annotate(str(1),xy=(1,4))
    
    plt.xticks(x, x_labels)
    xtickNames = ax.get_xticklabels()
    plt.setp(xtickNames, rotation=0, fontsize=10)
    #plt.ylabel('CIDEr', rotation=0)
    
    #ax.xaxis.set_label_coords(1.05, -0.025)
    #ax.legend( (lines[0][0], lines[1][0], lines[2][0], lines[3][0], lines[4][0]), \
    #    tuple(legend), ncol=2, fontsize=11,loc='upper left')
    
    ax.legend( (lines[0][0], lines[1][0], lines[2][0], lines[3][0]), \
        tuple(legend), ncol=2, fontsize=11,loc='upper left')
    
    #plt.savefig('exp1.png',bbox_inches='tight', pad_inches=0)

    plt.show()

    
def draw_history(input_dir):
    models = ['XE2', 'CST_GT_None', 'SCST', 'CST_MS_Greedy', 'CST_MS_HCB', 'CST_MS_MCB']
    legend = ['XE', 'CST_GT_None', 'SCST', 'CST_MS_Greedy', 'CST_MS_SCB', 'CST_MS_SCB(*)']
    
    x_list = []
    y_list = []
    for m in models:
        data_file = os.path.join(input_dir, m + '_history.json')
        data = json.load(open(data_file))
        data = [(int(k), 100*v['CIDEr']) for k,v in data.iteritems()]
        data = sorted(data, key=lambda x: x[0])
        epochs, scores = zip(*data)
        x_list.append(list(epochs))
        y_list.append(list(scores))
    
    #import pdb; pdb.set_trace()
    #x = range(1, len(y_list[0])+1)
    x_labels = range(0,200)
    
    fig = plt.figure( figsize=(6, 4), dpi=120)
    ax = fig.add_subplot(111)
    
    color_list = ['k', 'm', 'b', 'g', 'c', 'r', 'y']
    marker_list = ['>', 'd', 'h', 's', '*', '.', 'v']
    
    lines = [None]*len(y_list)
    for idx,y in enumerate(y_list):
        #lines[idx] = ax.plot(x_list[idx], y, '-', linewidth=2, marker=marker_list[idx], color=color_list[idx])
        lines[idx] = ax.plot(x_list[idx], y, '-', linewidth=1, color=color_list[idx])

    plt.grid()
    
    #ax.set_xlim(0.5, len(x)+ 0.5)
    #ax.set_ylim(0.25, 0.8)
    #ax.set_ylabel('CIDEr')
    ax.yaxis.set_label_coords(-0.04, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    ax.set_xlabel('Epoch')
    
    #ax.annotate(str(1),xy=(1,4))
    
    #plt.xticks(x, x_labels)
    xtickNames = ax.get_xticklabels()
    plt.setp(xtickNames, rotation=0, fontsize=10)
    plt.ylabel('CIDEr', rotation=0)
    
    #ax.xaxis.set_label_coords(1.05, -0.025)
    #ax.legend( (lines[0][0], lines[1][0], lines[2][0], lines[3][0], lines[4][0]), \
    #    tuple(legend), ncol=2, fontsize=11,loc='upper left')
    
    ax.legend((lines[0][0], lines[1][0], lines[2][0], lines[3][0], lines[4][0], lines[5][0]), \
        tuple(legend), ncol=3, fontsize=11, loc='lower center')
    
    #plt.savefig('exp1.png',bbox_inches='tight', pad_inches=0)

    plt.show()
    
if __name__ == '__main__':
    start = datetime.now()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    argparser = argparse.ArgumentParser(description = "Combine training data into one single file of each type")
    argparser.add_argument("graph", default='cider', choices=['cider', 'logp', 'history'], type=str)
    argparser.add_argument("--input_dir", default='output/model/cvpr2018_results', type=str)
    
    args = argparser.parse_args()
    
    if args.graph == 'cider':
        draw_cider(args.input_dir)
    elif args.graph == 'logp':
        draw_logp(args.input_dir)
    elif args.graph == 'history':    
        draw_history(args.input_dir)
    
    logger.info('done')
    logger.info('Time: {}'.format(datetime.now() - start))
