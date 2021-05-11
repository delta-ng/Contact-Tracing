from torchreid.utils import FeatureExtractor
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
import numpy as np
import cv2
import os
import random
import math


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
# print(len(folders))
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
print(match)
for i in range(len(match)-1,-1,-1):
    min_match=0.5 # [ Max value , Max index ]
    if max(match[i])>min_match:
        os.rename('past_ppl/'+str(i),'past_ppl/'+str(match[i].index(max(match[i]))))