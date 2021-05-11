# from torchreid.utils import FeatureExtractor
# from sklearn.metrics.pairwise import cosine_similarity
# from numpy import dot
# from numpy.linalg import norm
import numpy as np
import cv2
import os
import random
import math
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt



def plot(l):
    df = pd.DataFrame(l, columns=['source', 'target', 'frame'])
    df['weight'] = 1
    fr = df.frame.unique()
    for i in range(len(fr)):
        G=0
        a = df[df['frame']==fr[i]]
        a.reset_index()
        print(a)
        G = nx.from_pandas_edgelist(a,source='source',target='target',edge_attr='weight')
        nx.draw(G, with_labels=True)
        plt.tight_layout()
        plt.savefig("frame"+str(i)+".png", format="PNG")
        plt.clf()

def iou(box1, box2):
    xa = max( box1[1] , box2[1] )
    ya = max( box1[0] , box2[0] )
    xb = min( box1[3] , box2[3] )
    yb = min( box1[2] , box2[2] )
    
    interArea = max(0, xb - xa ) * max(0, yb - ya )

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1] )
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1] )
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = float(interArea) / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou


def detect_people(frame,personIdx=0):
	# grab the dimensions of the frame and  initialize the list of
	# results

	(H, W) = frame.shape[:2]
	results = []
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	boxes = []
	centroids = []
	confidences = []
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
	if len(idxs) > 0:

		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)

	return results

cap = cv2.VideoCapture('./3dcv (1).mp4')
MIN_CONF = 0.3
NMS_THRESH = 0.3
LABELS = open('yolo-coco/coco.names').read().strip().split("\n")

weightsPath = 'yolo-coco/yolov3.weights'
configPath = 'yolo-coco/yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
prev = []
id = []
prev_id = []
j = 0
past_ppl = './past_ppl'
iou_threshold = 0.50
initial = True 
count = 1
violation = []

while True: 
    r, img = cap.read()
    if not r:
        break
    boxes = detect_people(img,personIdx=LABELS.index("person"))
    # cordinates.append(boxes)
    curr_id =[]
    for i in range(len(boxes)):
        startX, startY, endX, endY = boxes[i][1]
        box = [startY,startX, endY, endX]
        cropped_img = img[ box[0]:box[2] , box[1]:box[3] ]
        cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
        done = False
        if initial:
            id.append(1)
            curr_id.append(len(id)-1)
            os.makedirs(past_ppl + '/' + str(i)  )
            cv2.imwrite(past_ppl + '/' + str(i) + '/1.jpg',cropped_img) 
        else:
            for index in range(len(prev)):
                # print(iou(prev[index][1],[]),box,prev[index][1])
                if iou(prev[index][1],boxes[i][1])>iou_threshold:
                    # print(prev_id[index],"Match")
                    id[prev_id[index]]+=1
                    curr_id.append(prev_id[index])
                    cv2.imwrite(past_ppl + '/' + str(prev_id[index]) + '/'+ str(id[prev_id[index]]) + '.jpg',cropped_img)
                    done = True
                    break
            if not(done):
                os.makedirs(past_ppl + '/' + str( len(id) )  )
                cv2.imwrite(past_ppl + '/' + str( len(id) )  + '/1.jpg',cropped_img)
                # cv2.imwrite('./temporaryImg.jpg',cropped_img)
                id.append(1)
                curr_id.append(len(id)-1)
    for i in range(len(boxes)):
        centroid=boxes[i][2]
        for j in range(i+1,len(boxes)):
            centroid1=boxes[j][2]
            dist=math.sqrt((centroid[0]-centroid1[0])**2+abs(centroid[1]-centroid1[1])**2)
            if dist<150:
                cv2.rectangle(img,(boxes[i][1][0],boxes[i][1][1]),(boxes[i][1][2],boxes[i][1][3]),(255,255,255),4)
                cv2.rectangle(img,(boxes[j][1][0],boxes[j][1][1]),(boxes[j][1][2],boxes[j][1][3]),(255,255,255),4)
                violation.append((curr_id[i],curr_id[j],count))
    cv2.imshow("preview", img)
    cv2.imwrite('Output/'+str(count)+'.jpg',img)
    prev = boxes
    prev_id = curr_id
    key = cv2.waitKey(1)
    initial = False 
    count+=1
    if key & 0xFF == ord('q'):
        break   
print(violation)