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

def generate_histogram_dict(anchorratios, 
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
    for ratio in anchorratios:
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
   
def get_anchors_ratios(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    anchorratios=[]
    benchmarkanchor={}
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            for anno in annotationdict[foldername][imagename]['annotations']:
                benchmarkanchor['class']=anno['label'].lower()
                benchmarkanchor['image']=imagename
                if anno['width'] and anno['height']:
                    benchmarkanchor['ratio']=anno['width']/anno['height']
                    anchorratios.append(benchmarkanchor)
                benchmarkanchor={}
                
    return anchorratios

def load_json_annotations(filepath, jsonlabel):
    
    
    annotationdict={}
    folderdict=os.listdir(filepath)
    for foldername in folderdict:
        jsonpath=os.path.join(filepath,foldername)
        # load the json files
        if not os.path.exists(os.path.join(jsonpath,'annotationfull_'+foldername+'.json')):
            continue   
        else:
            annotationdict[foldername]={}
            annotationdict[foldername]=json.load(open(os.path.join(jsonpath,'annotationfull_'+foldername+'.json')))
    
    return annotationdict

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 summer/Leading Vehicle/Viewnyx dataset/EventFrames', 
                        help="File path of input data")
    parser.add_argument('--json_label', type=str, 
                        default='with_leading', 
                        help="label to specify json file")
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get anchor ratios
    anchorratios = get_anchors_ratios(annotationdict)      
    
    # generate histogram (bins for different aspect ratios)
    hist, mean, std = generate_histogram_dict(anchorratios)
    
    # generate aspect ratios for the training
    aspectratios = generate_aspect_ratios(mean, std)      

    print('suggested anchor aspect ratios:\n',aspectratios)

""" End of File """