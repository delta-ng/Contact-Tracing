from torchreid.utils import FeatureExtractor
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import numpy as np
import cv2
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from redundancy import *

def ploting(l):
    df = pd.DataFrame(l, columns=['source', 'target', 'frame'])
    df['weight'] = 10
    fr = df.frame.unique()
    # plt.rcParams["figure.figsize"] = (5,5)
    for i in range(len(fr)):
        G=0
        a = df[df['frame']<=fr[i]]
        a.reset_index()
        print(a)
        G = nx.from_pandas_edgelist(a,source='source',target='target',edge_attr='weight')
        # nx.draw(G, with_labels=True)
        # random_pos = nx.random_layout(G, seed=0)#  
        random_pos = nx.spring_layout(G,scale=0.5,seed=0)
        # pos = nx.nx_pydot.graphviz_layout(G)  
        plt.rcParams["figure.figsize"] = (8,8)
        nx.draw(G,pos =random_pos,node_size=1500, node_color='lightblue', font_size=10, font_weight='bold',with_labels=True,width=2,edge_color='black')
        # plt.tight_layout()
        
        plt.savefig("./frame"+str(i)+".png", format="PNG")
        plt.clf()

ploting([(0, 2, 10), (0, 2, 11), (0, 2, 12), (0, 2, 13), (2, 1, 14), (0, 2, 17), (2, 0, 18), (2, 4, 19), (2, 0, 19), (4, 2, 20), (2, 1, 21), (2, 1, 22), (2, 4, 23), (4, 2, 24), (4, 2, 25), (3, 2, 30), (3, 2, 31), (1, 2, 45), (1, 2, 46), (2, 1, 47), (1, 2, 48), (2, 1, 49), (1, 2, 50), (1, 2, 51), (1, 2, 64), (2, 1, 65), (2, 1, 66), (2, 1, 67), (1, 2, 68), (0, 3, 71), (0, 3, 72), (0, 3, 73)])