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
                                 rejsize_h=15,
                                 rejsize_w=15,
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
                        benchmarkanchor['diag']=math.sqrt(anno['width']*anno['height'])
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

def getGridFromHist(hist, threshlist=None, pel_list=None, inverse=True,rejectsize=31):
    """
    Covert the Histogram into Probability Density Function (PDF), set a few 
    thresholds for Cumulative Distribution Function (CDF), and return corresponding
    .
    
    """
    for i in range(len(hist[0])):
        if hist[1][i]<rejectsize:
            hist[0][i]=0
    total=np.sum(hist[0])# hist[0] are numbers in bins, hist[1] are the setup of bins
    count=0
    
    if inverse:
        """
        input:
            hist, pel_list
        output:
            thresh_list
        
        """
        thresh_list=[]
        for pel_thresh in pel_list:
            for i in range(len(hist[0])):
                count+=hist[0][i]
                pel=hist[1][i]
                if pel > pel_thresh:
                    thresh_list.append(float(count)/total)
                    count=0
                    break
        return thresh_list
    else:
        """
        input:
            hist, thresh_list
        output:
            pel_list
        
        """
        pel_list=[]
        for thresh in threshlist:
            for i in range(len(hist[0])):
                count+=hist[0][i]
                percent=float(count)/total
                if percent>=thresh:
                    pel_list.append(hist[1][i])
                    count=0
                    break
        return pel_list

def drawCDF(pel_list, hist_thresh_list):
    chartaxis = [0.0,800.0,0.0,1.0]
    plt.figure(figsize=(4,4),dpi=100)
    plt.axis(chartaxis)
    plt.plot(pel_list, hist_thresh_list,'bo-')
    plt.title('CDF over pixel size of bbox')
    plt.xlabel('pixel size')
    plt.ylabel('probability')
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k', 
                        help="File path of input data")
    parser.add_argument('--json_label', type=str, 
                        default='val_VIVA_format_crop_gt22.json', 
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

    print('suggested Gaussian anchor aspect ratios:\n',aspectratios)
    
    # generate histogram of bbox size
    h_hist, w_hist, diag_hist = generate_bbox_size_histogram(bboxes)
    
# =============================================================================
#     # get grid size from bbox
#     #hist_thresh_list=[0.15,0.3,0.6,0.9]
#     hist_thresh_list=[0.05,0.1,0.15,0.2,0.25,
#                       0.3,0.35,0.4,0.5,0.6,
#                       0.7,0.8,0.9,0.99,1.0]
#     pel_list = getGridFromHist(diag_hist, threshlist=hist_thresh_list, 
#                                       pel_list=None, inverse=False)
#     grid_list = [ round(800.0/i) for i in pel_list ]
#     drawCDF(pel_list, hist_thresh_list)
# =============================================================================
    
    
    # bbox numbers for grid sizes
    gridlist=[19,10,5,3,2,1] 
    # 19 derive from rejected size, 19,10,5 from distribution (thresh 30%,60%,90%),
    # 19 repeated thus removed, 3,2,1 added for improving
    # the capability of detecting near vehicles even the number of bbox for these
    # two grid sizes are few
    h_pel_list=[800/i for i in gridlist]
    percentage_list = getGridFromHist(diag_hist, threshlist=None, 
                                      pel_list=h_pel_list, inverse=True)
    
        
""" End of File """