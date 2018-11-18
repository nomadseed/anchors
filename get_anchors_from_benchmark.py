# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 15:38:30 2018

Check bbx size in all the benchmarks ( and draw histogram ) and automatically 
generate certain numbers of anchors according to the histogram

@author: Wen Wen
"""

import json
import os
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


def gaussian_comparison():
    n_groups = 5

    means_men = (20, 35, 30, 35, 27)
    std_men = (2, 3, 4, 1, 2)
    
    means_women = (25, 32, 34, 20, 25)
    std_women = (3, 5, 2, 3, 3)
    
    fig, ax = plt.subplots()
    
    index = np.arange(n_groups)
    bar_width = 0.35
    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    
    rects1 = ax.bar(index, means_men, bar_width,
                    alpha=opacity, color='b',
                    yerr=std_men, error_kw=error_config,
                    label='Men')
    
    rects2 = ax.bar(index + bar_width, means_women, bar_width,
                    alpha=opacity, color='r',
                    yerr=std_women, error_kw=error_config,
                    label='Women')
    
    ax.set_xlabel('Group')
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
    ax.legend()
    
    fig.tight_layout()
    plt.show()
    return fig

def generate_aspect_ratios(mean, std):
    aspectratios=[]
    for i in range(-2,3):
        # take -2 std to +2 std for it contains up to 95.4% of all aspect ratios
        aspectratios.append(round(mean+i*std,2))
    return aspectratios

def generate_histogram_dict(bboxes, 
                            numbins=200, 
                            minratio=0.3, 
                            maxratio=3.0,
                            debug=True):
    '''
    
    
    ^
    |            # # #
    |           # # # #
    |          # # # # #  
    |        # # # # # # # 
    |  # # # # # # # # # # # # # 
    -------------------------------->
    0.3333         1           3.0  aspect ratio (w/h)
    '''
    # build histogram
    ratiolist=[]
    for ratio in bboxes:
        ratiolist.append(ratio['ratio'])                
    hist=np.histogram(ratiolist,numbins,(minratio,maxratio))
    
    # calculate mean and variance
    mean=float(round(np.array(ratiolist).mean(),2))
    std=float(round(np.array(ratiolist).std(),2))
    
    # plot the chart for debug
    if debug:
        plt.hist(ratiolist,numbins,(minratio,maxratio)) 
        plt.title('Aspect Ratio Histogram')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('BBox count')
        plt.show()   
    
    return hist, mean, std

def generate_bbox_size_histogram(bboxes, 
                                 numbins=200,
                                 rejsize_h=22,
                                 rejsize_w=22,
                                 debug=True):
    '''
    
    
    ^
    |     #
    |    # # 
    |   # # # #  
    |  # # # # # #  
    | # # # # # # # # # # # # # 
    -------------------------------->
    0                          maxsize  (w or h)
    '''
    hlist=[]
    wlist=[]
    diaglist=[]
    totalbox=len(bboxes)
    wreject=0
    hreject=0
    allreject=0
    for bbox in bboxes:
        hlist.append(bbox['h'])
        wlist.append(bbox['w'])
        diaglist.append(bbox['diag'])
        if bbox['h']<rejsize_h:
            hreject+=1
        if bbox['w']<rejsize_w:
            wreject+=1
        if bbox['w']<rejsize_w and bbox['h']<rejsize_h:
            allreject+=1
    h_hist=np.histogram(hlist,numbins,(0,480))
    w_hist=np.histogram(wlist,numbins,(0,640))
    diag_hist=np.histogram(diaglist,numbins,(0,800))
    
    # plot the chart for debug
    if debug:
        plt.hist(hlist,numbins,(0,480)) 
        plt.title('bbox height Histogram')
        plt.xlabel('pixel size')
        plt.ylabel('BBox count')
        plt.show()
        
        plt.hist(wlist,numbins,(0,640)) 
        plt.title('bbox width Histogram')
        plt.xlabel('pixel size')
        plt.ylabel('BBox count')
        plt.show()
        
        plt.hist(wlist,numbins,(0,800)) 
        plt.title('bbox diagonal Histogram')
        plt.xlabel('pixel size')
        plt.ylabel('BBox count')
        plt.show()
    
    # print the percent of rejected boxes
    print('{}% bbox rejected by width'.format(wreject/totalbox*100))
    print('{}% bbox rejected by height'.format(hreject/totalbox*100))
    print('{}% bbox rejected by both width and height'.format(allreject/totalbox*100))
    
    
    return h_hist,w_hist, diag_hist


def get_bboxes(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    bboxes=[]
    benchmarkanchor={}
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            if len(annotationdict[foldername][imagename])==0:
                continue
            else:
                for anno in annotationdict[foldername][imagename]['annotations']:
                    benchmarkanchor['class']=anno['label'].lower()
                    benchmarkanchor['image']=imagename
                    if anno['width'] and anno['height']:
                        benchmarkanchor['ratio']=anno['width']/anno['height']
                        benchmarkanchor['w']=anno['width']
                        benchmarkanchor['h']=anno['height']
                        benchmarkanchor['diag']=math.sqrt(anno['width']*anno['width']+anno['height']*anno['height'])
                        bboxes.append(benchmarkanchor)
                    benchmarkanchor={}
                
    return bboxes

def load_json_annotations(filepath, jsonlabel):
    
    
    annotationdict={}
    folderdict=os.listdir(filepath)
    for foldername in folderdict:
        jsonpath=os.path.join(filepath,foldername)
        if '.' in jsonpath:
            continue
        # load the json files
        jsondict=os.listdir(jsonpath)
        for jsonname in jsondict:
            if jsonlabel in jsonname and '.json'in jsonname:
                annotationdict[foldername]={}
                annotationdict[foldername]=json.load(open(os.path.join(jsonpath,jsonname)))
    
    return annotationdict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/debug/bdd100k', 
                        help="File path of input data")
    parser.add_argument('--json_label', type=str, 
                        default='crop', 
                        help="label to specify json file")
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get anchor ratios
    bboxes = get_bboxes(annotationdict)      
    
    # generate histogram (bins for different aspect ratios)
    hist, mean, std = generate_histogram_dict(bboxes)
    
    # generate aspect ratios for the training
    aspectratios = generate_aspect_ratios(mean, std)      

    print('suggested anchor aspect ratios:\n',aspectratios)
    
    # generate histogram of bbox size
    h_hist, w_hist, _ = generate_bbox_size_histogram(bboxes)
    
    
""" End of File """