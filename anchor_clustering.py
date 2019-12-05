# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:21:58 2019

clustering for getting optimized anchors

methods:
    k-means

@author: Wen Wen
"""
from sklearn.cluster import KMeans
import numpy as np
import time
import os
import json
import argparse
import matplotlib.pyplot as plt

savedKmeanList={'ssd_opt_gt22':[
                    [16,13],[23,15],[18,20],
                    [25,20],[50,23],[33,38],
                    [51,47],[68,68],[43,65],[110,34],
                    [115,107],[153,63],[99,78],[78,101],
                    [146,126],[200,161],[216,96],[128,170],
                    [221,230],[180,280],[291,174],[271,260],[251,195]                    
                    ],
                'ssd_clust23p':[
                    [6,6],
                    [13,18],[10,11],
                    [27,33],[20,29],[22,17],
                    [51,33],[20,71],[35,41],
                    [52,52],[77,55],[22,114],[60,71],
                    [84,84],[57,159],[134,120],[113,165],[63,109],[77,236],
                            [97,120],[161,231],[178,153],[112,83]
                    ],
                'ssd_clust23p_gt22':[
                    [15,17],
                    [20,26],[30,36],[29,22],[44,30],
                    [59,65],[61,44],[83,57],[43,48],[26,79],
                    [81,80],[34,130],[61,100],[93,105],[80,139],[115,83],
                    [65,180],[80,250],[160,125],[203,171],[155,246],
                        [131,175],[119,128]
                    ]
                }


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

def returnBoxArea(box):
    # return area of box
    return box[0]**2+box[1]**2

def returnBoxEdge(box):
    # return area of box
    return box[0]+box[1]

def get_bboxes(annotationdict):
    '''
    
    anchorratios=[{'class':'car',
                   'image':'xxx.jpg',
                   'ratio':0.5}, {}, {} ]
    '''
    bboxlist=[]
    for foldername in annotationdict:
        for imagename in annotationdict[foldername]:
            if len(annotationdict[foldername][imagename])==0:
                continue
            else:
                for anno in annotationdict[foldername][imagename]['annotations']:
                    if anno['width'] and anno['height']:
                        bboxlist.append([round(anno['width']*0.46875),round(anno['height']*0.625)])
                
    return bboxlist

def groupBBox(bboxlist,threshlist,groupshape):
    """
    group bbox into different groups by grid size, save all the bbox into a dict
    that contain multiple bbox lists
    
    """
    xlist=bboxlist.copy()
    bboxdict={}
    print('grouping bbox based on threshlist, method: '+groupshape)
    if groupshape=='sector':
        xlist.sort(key=returnBoxArea,reverse=True)
        for thresh in threshlist:
            curlist=[]
            for i in range(len(xlist)):
                box=xlist.pop()    
                if box[0]**2+box[1]**2<thresh**2:
                    curlist.append(box)
                else:
                    break
        
            bboxdict[thresh]=curlist
    elif groupshape=='belt':
        xlist.sort(key=returnBoxEdge,reverse=True)
        for thresh in threshlist:
            curlist=[]
            for i in range(len(xlist)):
                box=xlist.pop()    
                if box[0]+box[1]<thresh:
                    curlist.append(box)
                else:
                    break
        
            bboxdict[thresh]=curlist
    else:
        raise ValueError('groupshape can only be belt or sector')
    return bboxdict

def simpleKMeans(xlist,n_cluster=2,random_state=0,algorithm='auto'):
    """
    
    output:
        centroids: centroids acquired from clustering
        labels: clustering result of input array
        score: sum of distances from centroids to all the points in their cluster
        time: processing time of current k-means clustering
        algorithm:
            'full' for classical EM-style algorithm
            'elkan' is more efficient by using the traiangle inequality, but 
                not suitable for sparse data.
            'auto' will choose elkan for dense data and full for sparse data
        
    """
    if len(xlist)==0:#skip empty list
        return [],[],1,0
    
    starttime=time.time()
    kmeans = KMeans(n_clusters=n_cluster, random_state=random_state,algorithm=algorithm).fit(xlist)
    processtime = time.time()-starttime
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score = -kmeans.score(xlist)
    
    return centroids, labels, score, processtime

def adaptiveKMeans(bboxdict,max_num_center=10,scorethresh=0.8):
    """
    take in a bboxlist and perform k-means clustering where k ranges from 2 to
    a maximum (defalut is 10). 
    Using Elbow method to get the optimized k value for the bboxlist, where the
    elbow point is defined by a score threshold. when current score is larger
    than last score, i.e. score>scorethresh*last_score, the last point is the
    elbow point. note that the changing threshold will have huge impact to the 
    final result.
    
    in the clusterlist, all the default boxes are saved as [width, height]
    
    """
    clusterlist=[]
    sumscore=0
    for thresh in bboxdict:
        grouplist=[]
        lastscore=float('Inf')
        for n_cluster in range(2,max_num_center):
            # k-means clustering        
            centroids, labels, score, processtime = simpleKMeans(xlist=bboxdict[thresh], n_cluster=n_cluster)
            print('clustering for k={}, processing time={}, score={}'.format(n_cluster,processtime,score))
            singledict={}
            singledict['k']=n_cluster
            if len(centroids)==0:
                singledict['centroids']=[]
                singledict['labels']=[]
            else:
                singledict['centroids']=centroids.tolist()
                singledict['labels']=labels.tolist()
            singledict['score']=score
            sumscore+=score
            singledict['processtime']=processtime
            grouplist.append(singledict)
            if score >= lastscore*scorethresh:
                print('current score is {}% of last score, stop clustering for current layer'.format(score/lastscore))
                break
            lastscore=score
        
        clusterlist.append({'thresh':thresh, 'final_centro':singledict['centroids'],'totalscore':sumscore})
    return clusterlist

def plotScatter(bboxdict, clusterdict, savepath='',samecolor=False):
    chartaxis = [0.0,310.0,0.0,310.0]
    plt.figure(figsize=(6,6),dpi=100)
    plt.axis(chartaxis)
    if samecolor:
        colormap=['#0000ff','#0000ff']
    else:
        colormap=['#0000ff', '#00ff00']# use html colors
    centrolist=[i['final_centro'] for i in clusterdict]

    ax = plt.subplot()
    
    # plot scatter for benchmarks
    for i, thresh in zip(range(len(bboxdict)),bboxdict):
        x_box = [i[0] for i in bboxdict[thresh]]
        y_box = [i[1] for i in bboxdict[thresh]]
        area_box = np.ones(len(x_box))*1
        ax.scatter(x_box, y_box, s=area_box, alpha=0.2, c=colormap[i%2], marker='o')# colormap[i]*len(x_box)
    

    # plot centroids
    for anchorlist in centrolist:
        x_anchor = [i[0] for i in anchorlist]
        y_anchor = [i[1] for i in anchorlist]
        area_anchor = np.ones(len(anchorlist))*100
        ax.scatter(x_anchor, y_anchor, s=area_anchor, alpha=1, c='r', marker='.')
    
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.title('Distribution of Groundtruth BBox and anchors')
    plt.grid(True)
    plt.savefig(os.path.join(savepath,'scatter.png'),dpi=100)
    print('scatter saved as\n {}'.format(os.path.join(savepath,'scatter.png')))
    plt.show()
    #plt.close()

def plotScatterSaved(bboxdict,clusterlist_1,clusterlist_2,
                     savepath):
    """
    this is a duplication of the function 'plotScatter', but take the saved
    list of kmeans centroids instead, this is a function for pure comparison
    
    """
    
    chartaxis = [0.0,310.0,0.0,310.0]
    plt.figure(figsize=(6,6),dpi=100)
    plt.axis(chartaxis)
    
    ax = plt.subplot(axisbg='black')
    
    # plot scatter for benchmarks
    for i, thresh in zip(range(len(bboxdict)),bboxdict):
        x_box = [i[0] for i in bboxdict[thresh]]
        y_box = [i[1] for i in bboxdict[thresh]]
        area_box = np.ones(len(x_box))*1
        ax.scatter(x_box, y_box, s=area_box, alpha=0.3, c='b', marker='o')
        
    # plot centroids of cluster result 1 in red
    for anchor in savedKmeanList[clusterlist_1]:
        x_anchor=anchor[1]
        y_anchor=anchor[0]
        ax.scatter(x_anchor, y_anchor, s=30, alpha=1, c='r', marker='o')
    
    # plot centroids of cluster result 2 in blue
    for anchor in savedKmeanList[clusterlist_2]:
        x_anchor=anchor[0]
        y_anchor=anchor[1]
        ax.scatter(x_anchor, y_anchor, s=30, alpha=1, c='y', marker='^')
    
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.title('Raw and Layer-wise K-means Clustering')
    #plt.grid(True)
    plt.savefig(os.path.join(savepath,'kmeans_comparison.png'),dpi=100)
    print('scatter saved as\n {}'.format(os.path.join(savepath,'kmeans_comparison.png')))
    print('red dots are {}, yellow triangles are {}'.format(clusterlist_1,
                                         clusterlist_2))
    plt.show()
    #plt.close()
    
  
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, 
                        default='D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k', 
                        help="File path of input data")
    #D:/Private Manager/Personal File/uOttawa/Lab works/2018 fall/BerkleyDeepDrive/bdd100k
    #D:/Private Manager/Personal File/uOttawa/Lab works/2019 summer/caltech_dataset
    
    parser.add_argument('--json_label', type=str, 
                        default='bdd100k_labels_images_train_VIVA_format_crop_gt22.json', 
                        help="label to specify json file")
    #bdd100k_labels_images_val_VIVA_format_crop.json
    #caltech_annotation_VIVA_test.json
    
    args = parser.parse_args()
    
    filepath = args.file_path
    jsonlabel = args.json_label
    savepath = os.path.join(filepath,'ssd_cluster_result')
    
    # get annotations of whole dataset
    annotationdict = load_json_annotations(filepath, jsonlabel)
    
    # get bboxlist from annotation
    bboxlist = get_bboxes(annotationdict)
    
    # setup threshlist, where threshold is diagnal length in pixel
    # to get the threshlist easily, check the get_anchor_from_bench_mark.py
    # run the file and check h_pel_list
    
    # SETUPS
    #threshtype='max'
    #threshtype='mean'
    threshtype='all'
    scorethresh=0.7 # default=0.7
    #groupshape='belt'
    groupshape='sector'
    compareonly=True
    
    
    '''threshes for various dataset'''
    #threshlist = [500] # yolo clustering
    if threshtype=='max':
        threshlist = [15.78,30,60,100,150,500] # Threshlist for BDD dataset, of grid size [19,10,5,3,2,1]
    elif threshtype=='mean':
        threshlist = [20,45,80,125,225,500] # Threshlist for BDD dataset, of grid size [19,10,5,3,2,1], but use grid size as centre
    #threshlist=[13.63, 17.64, 33.33, 75, 150, 500] # Threshlist for Caltech dataset, of grid size [22,17,9,4]
    elif threshtype=='all':
        threshlist=[10000]
        scorethresh=1
    
    bboxdict = groupBBox(bboxlist,threshlist,groupshape=groupshape)
    
    # perform k-means for all the layers
    if not compareonly:
        clusterdict = adaptiveKMeans(bboxdict,max_num_center=24,scorethresh=scorethresh)#max_num_center=25
        json.dump(clusterdict, open(os.path.join(savepath,'k-means_cluster_result_ssd.json'), 'w'),sort_keys=True, indent=4)

    # draw figures of [sum of error over k values]
    if not compareonly:
        plotScatter(bboxdict, clusterdict, savepath=savepath,samecolor=False)
        
        # check IOU error after clustering
        scorelist=[i['totalscore'] for i in clusterdict]
        layer=np.array([19*19,10*10,5*5,3*3,2*2,1*1])
        centro=np.array([len(i['final_centro']) for i in clusterdict])
        priors=np.sum(layer*centro)
        centronum=sum(centro)
        score=sum(scorelist)
        avgscore=score/centronum
        print('avgscore: {}'.format(avgscore))
        print('priors: {}'.format(priors))
    else:
        plotScatterSaved(bboxdict,
                         clusterlist_1='ssd_opt_gt22',
                         clusterlist_2='ssd_clust23p_gt22',
                         savepath=savepath)

    


"""End of file"""