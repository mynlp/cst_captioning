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
from datetime import datetime

logger = logging.getLogger(__name__)

import numpy as np

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import matplotlib.ticker as ticker

def load_data(input_file):
    y_list = []
    legend = []
    with open(input_file, 'rb') as f:
        reader = csv.reader(f)
        for i,row in enumerate(reader):
            if i == 0:
                legend = row[5:]
            elif i == 1:
                continue
            else:
                y_data = [float(r) for r in row[5:]]
                y_list.append(y_data)
    y_list = [list(x) for x in zip(*y_list)]            
    return y_list, legend
    
if __name__ == '__main__':
    start = datetime.now()

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(levelname)s: %(message)s')
    argparser = argparse.ArgumentParser(description = "Combine training data into one single file of each type")
    argparser.add_argument("--input_dir", default='output/model/cvpr2018_results', type=str)
    
    args = argparser.parse_args()
    
    legend = ['XE', 'CST_XE', 'CST_XE_SCB', 'CST_RL_Greedy', 'CST_RL_SCB']
    
    y_list = []
    for m in legend:
        data_file = os.path.join(args.input_dir, m + '_dataslice.txt')
        lines = [float(line) for line in open(data_file)]
        y_list.append(lines)
    
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
    ax.legend( (lines[0][0], lines[1][0], lines[2][0], lines[3][0], lines[4][0]), \
        tuple(legend), ncol=2, fontsize=11,loc='upper left')
    
    #plt.savefig('exp1.png',bbox_inches='tight', pad_inches=0)

    plt.show()
    logger.info('done')
    logger.info('Time: {}'.format(datetime.now() - start))
