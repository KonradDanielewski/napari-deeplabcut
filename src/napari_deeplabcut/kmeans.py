from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd 
from itertools import *


def find_y_size(df, filenames, num, scorer, bodyparts):
    # function for finding the height of the horse in the selected image
    
    y_min = min(df.loc[filenames[num], (scorer,bodyparts,'y' )])
    y_max = max(df.loc[filenames[num], (scorer,bodyparts,'y' )])
    
    return (y_max - y_min)

def resizing_images(samp_size, df_features, filenames, distances, df, scorer,bodyparts):
    # resizing the images that all horses are about the same size

    y_size = [] # height of the horse in the image
    coef = [] # coefficient: height of the horse in the image divided by height of the horse in selected image
    for i in range(df_features.shape[0]):
        y_size.append(find_y_size(df, filenames, i, scorer, bodyparts))
        coef.append(y_size[i]/samp_size)
    df_norm = df_features.copy()
    for i in range(df_features.shape[1]):
        df_norm.loc[filenames,distances[i]] = [a/b for a,b in zip(df_features.loc[filenames,distances[i]], coef)]
    
    return df_norm

def comp_dist(features,df, scorer, bodypart1, bodypart2):
    # function for computing the distance between two body parts: bodypart1 and bodypart2
    
    name = bodypart1 + '_' + bodypart2
    features[name] = np.sqrt(np.square(df[scorer,bodypart1, 'x'] - df[scorer,bodypart2, 'x'])+np.square(df[scorer,bodypart1, 'y'] - df[scorer, bodypart2, 'y']))
    
    return features

def cluster(resized_scaled_data):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(resized_scaled_data)

    # putting components in a dataframe for later
    PCA_components = pd.DataFrame(principalComponents)

    dbscan=DBSCAN(eps = 1.4, min_samples = 20)

    # fit - perform DBSCAN clustering from features, or distance matrix.
    dbscan = dbscan.fit(PCA_components)
    cluster1 = dbscan.labels_

    return PCA_components, cluster1

def to_plot(X,cluster):
    df = X
    df.loc[:,'label'] = cluster
    colors = {-1: 'red', 0: 'blue', 1:'orange', 2:'green', 3:'yellow', 4:'black', 5:'gold', 6:'lightblue', 7:'darkgreen'}
    return df, colors #check!


def read_data(url):
    header = [0,1,2] #change if ma
    df = pd.read_hdf(url)
    df = df.dropna()
    #print(df.head())
    filenames = df.index # names of images
    scorer = 'Byron'
    #print(df)
    bodyparts = np.zeros(len(df.columns)).astype(str)
    coord = np.zeros(len(df.columns)).astype(str)
    a = df.columns
    for i in range(len(df.columns)):
        bodyparts[i] = a[i][1]
        coord[i] = a[i][2]
    
    bodyparts = np.unique(bodyparts) # 22 unique labels
    #print("Unique body parts:",bodyparts)

    #inx = np.where(filenames == 'BrownHorseinShadow/0135.png')
    sample =df.loc['BrownHorseinShadow/0135.png', (scorer, bodyparts, 'y')]
    samp_size = max(sample) - min(sample)
    print(samp_size)
    features = pd.DataFrame()

    for bodypart_list in combinations(bodyparts, 2):
        features = comp_dist(features,df, scorer,bodypart_list[0], bodypart_list[1])
    distances = features.columns

    resized_features = resizing_images(samp_size, features, filenames, distances, df, scorer,bodyparts)
    names = resized_features.index
    PCA_components, cluster1 = cluster(resized_features)
    point_c , color = to_plot(PCA_components,cluster1)
    return point_c , color ,names 