from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
import cv2
from PIL import Image

import more_itertools as mit

def boundary_extraction(frame_list, nn):
    #리스트 내에 연속된 프레임들끼리 묶기
    final_list = []
    for i in frame_list:
        i.sort()
        ll = len(i)
        groups = [list(groups) for groups in mit.consecutive_groups(i)]

        # 덩어리 내에 소수의 누락 프레임 고려, 합치기
        add_list = []
        for i in range(len(groups) - 1):
            if (groups[i + 1][0] - groups[i][-1]) < int(ll * nn): # nn은 사용자 지정
                add_list.append(i)

        for n, i in enumerate(add_list):
            groups[i-n] = groups[i-n] + groups[i-n+1]
            del groups[i-n+1]

        #가장 큰 덩어리 바운더리 추출
        lens = []
        for i in range(len(groups)):
            lens.append(len(groups[i]))
        final_list.append(min(groups[lens.index(max(lens))]))
        final_list.append(max(groups[lens.index(max(lens))]))

    final_list.sort()
    return int((final_list[1]+final_list[2])/2), int((final_list[3]+final_list[4])/2)


#https://towardsdatascience.com/image-clustering-using-k-means-4a78478d2b83
# Function to Extract features from the images
def image_feature(originalFeatures):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];

    #vidcap = cv2.VideoCapture(vidFilePath)
    #success = True 
    #i = 0 
   # while(success): 
    for feat in originalFeatures : 
        #print(i)
        #succ, np_img = vidcap.read()
        #print(succ, np_img)
        #if succ == False : 
        #    break
        np_img = feat[1]
        idx = feat[0]

        img = Image.fromarray(np_img)
        img = img.resize((224,224), Image.ANTIALIAS)
        x = np.array(img)

        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        feat=model.predict(x)
        feat=feat.flatten()
        features.append(feat)
        img_name.append(idx)
        #i += 1 
    return features,img_name

#img_path=os.listdir('data')
def featureKmeans(originalFeatures,q2):
#vidFilePath = ["./testVideo/C04/C04_062.mp4"]


    img_features,img_name=image_feature(originalFeatures)

    #print(len(img_features))
    #print(img_features)

    #Creating Clusters
    startFrame = 0 
    #for ratio in template_ratio : 
    k = 3
    clusters = KMeans(k, random_state = 40)
    #clusters.fit(img_features[0 : int(startFrame + (len(img_features)*ratio))])
    clusters.fit(img_features)
    image_cluster = pd.DataFrame(img_name,columns=['image'])
    image_cluster["clusterid"] = clusters.labels_

    #print(image_cluster['clusterid'])
    print(len(image_cluster))
    # Images will be seperated according to cluster they belong
    clus_list = [[], [], []]
    for i in range(len(image_cluster)):
        if image_cluster['clusterid'][i]==0:
            #shutil.move(os.path.join('data', image_cluster['image'][i]), '0act')
            clus_list[0].append(image_cluster['image'][i])
        elif image_cluster['clusterid'][i]==1:
            clus_list[1].append(image_cluster['image'][i])
            #shutil.move(os.path.join('data', image_cluster['image'][i]), '1act')
        elif image_cluster['clusterid'][i]==2:
            clus_list[2].append(image_cluster['image'][i])
            #shutil.move(os.path.join('data', image_cluster['image'][i]), '2act')

    print(clus_list)
    b1, b2 = boundary_extraction(clus_list, k)
    print(b1,b2)
    #return b1, b2 
    q2.append([b1,b2])