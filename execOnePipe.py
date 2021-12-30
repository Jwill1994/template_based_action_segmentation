from video_Kmeans import * 
from video_lstmAL import * 
import multiprocessing
import threading
import time 
import cv2 
from copy import deepcopy 
t1 = time.time()

inputPath = "./PKBT_Press/C6918_W24_Recut_handle/C6918_W24_Recut_handle_01.mp4"
templateRatio = [0.3, 0.3, 0.4]


success = True 
i = 0
originalFeatures = []
vidcap = cv2.VideoCapture(inputPath)
while(success): 
	success, np_img = vidcap.read()
	if success == False : break 
	originalFeatures.append([i, np_img])
	i += 1 
print('Video to features Finished')

final_output = [] 
start_point = 0 
print(len(originalFeatures))
for fin_index in range(len(templateRatio)-1):
	fin_index = fin_index - 1
	fin_point = int(len(originalFeatures) * sum(templateRatio[:fin_index + 2])) #fin_index + 1 : start from 1
	#print(lenFirstTwoSegs)
	twoSegFeatures = originalFeatures[start_point : fin_point]
	#deq_input = [i[1] for i in twoSegFeatures]
	print('start,fin',start_point, fin_point)
	global q1, q2
	q1 = [] 
	q2 = [] 
	thread1 = threading.Thread(target=lstmAL,  args=(twoSegFeatures,q1))
	thread2 = threading.Thread(target=featureKmeans, args=(twoSegFeatures,q2))

	thread1.start()
	thread2.start()

	th1 = thread1.join()
	th2 = thread2.join()
	print(type(q1[0]),type(q2[0]))
	candidates = deepcopy(q2[0])
	if len(q1) is not 0 : 
		for i in range(len(q1)):
			candidates.append(q1[i])
	print('candidates boundaries from lstm, kmeans algorithms:',candidates)

	gt_boundary = fin_point  #template 비율로 짜른 처음 경계
	print('gt boundary by ratio', gt_boundary)
	diff_list = [abs(i - gt_boundary) for i in candidates]
	pred_boundary = candidates[diff_list.index(min(diff_list))] #알고리즘들로 자른 경계 중 비율로 자른 경계와 가까운 놈 

	start_point = pred_boundary
	print('Final pred boundary of this two segments', pred_boundary) 
	t2 = time.time()

	print('computation time', t2-t1)
	final_output.append(pred_boundary)
print(final_output)