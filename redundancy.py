from torchreid.utils import FeatureExtractor
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import numpy as np
import cv2
import os
import random
import math



# print(len(folders))

# Reference : https://www.geeksforgeeks.org/connected-components-in-an-undirected-graph/
class Graph:
 
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for i in range(V)]
 
    def DFSUtil(self, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in self.adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp
 
    # method to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)
 
    # Method to retrieve connected components
    # in an undirected graph
    def connectedComponents(self):
        visited = []
        cc = []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if visited[v] == False:
                temp = []
                cc.append(self.DFSUtil(temp, v, visited))
        return cc

def check_duplicate():
    past_ppl = './past_ppl'
    folders = os.listdir(past_ppl)
    image_list = []
    folder_no = []

    extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='model/osnet_ms_d_c.pth.tar',
            device='cpu'
        )
    prev=0
    for folder in range(len(folders)):
        files = os.listdir(past_ppl + '/' + str(folder))
        # Can choose random 5 images and do this.
        done = {}
        for _ in range(min(10,len(files))):
            i=random.choice([j for j in range(min(10,len(files)))])
            while i in done:
                i=random.choice([j for j in range(min(10,len(files)))])
            done[i]=True
            image_list.append('past_ppl/'+str(folder)+'/'+str(i+1)+'.jpg')
        folder_no.append([prev,len(image_list)])
        prev=len(image_list)
    # print(image_list)
    features = extractor(image_list)
    features = np.array(features)
    match=[[0 for i in range(len(folders))] for i in range(len(folders))]
    for i in range(len(folders)):
        elements = range(folder_no[i][0],folder_no[i][1])
        for k1 in elements:
            # done = False
            for j in range(len(folders)):
                if(i!=j):
                    t=0
                    elements1 = range(folder_no[j][0],folder_no[j][1])
                    for k2 in elements1:
                        if(dot(features[k1],features[k2])/(norm(features[k1])*norm(features[k2])) > 0.8):
                            t+=1
                    if t>0.6*len(elements1):
                        match[i][j]+=1
        match[i]=[match[i][t]/len(elements) for t in range(len(match[i]))]
        # break
    # print(match)
    g = Graph(len(folders))
    for i in range(len(match)-1,-1,-1):
        min_match=0.5 # [ Max value , Max index ]
        if max(match[i])>min_match:
            g.addEdge(i,match[i].index(max(match[i])))
    cc = g.connectedComponents()
    dict_cc={}
    for i in range(len(cc)):
        for j in cc[i]:
            dict_cc[j]=i
    return dict_cc
        # os.rename('past_ppl/'+str(i),'past_ppl/'+str(match[i].index(max(match[i]))))