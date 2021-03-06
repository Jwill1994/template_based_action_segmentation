from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes

import numpy as np
import os
from os import listdir
from os.path import isfile, join, isdir
from random import shuffle, choice
from PIL import Image
import sys
import json
import collections, cv2
from PIL import Image
from scipy import signal
from math import factorial
from collections import Counter
from math import log
#from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings
#https://github.com/CVPRUSFTampa/EventSegmentation
#batch = ["./testVideo/C04/C04_062.mp4"]

input_width = 224
input_height = 224
num_channels = 3
slim = tf.contrib.slim
n_hidden1 = 4096
n_hidden2 = 4096
feature_size = 4096
learnError = 0
n_epochs = 1
batch_size = 2
min_steps = batch_size

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

#originalFeatures : [ [idx, np_img], [], [] .. ]
def loadMiniBatch(originalFeatures): 

	#vidcap = cv2.VideoCapture(vidFilePath) ################
	#print(originalFeatures[0])
	#exit()
	deq_input = [i[1] for i in originalFeatures] # only np_img list 
	minibatch = []

	deq = collections.deque([deq_input[0]]) # put first frame 
	'''
	if deq[0][0] == False :  
		print("frame read failed")
		exit()
	'''
	#while success : 
	for _np_img in deq_input[1:] : 	
		deq.append(_np_img) 
		'''
		if deq[-1][0] == False :
			print("frame read failed or video finished!")
			success = False 
			break
		'''
		im = []
		#numFrames = 0
		for np_img in deq : 
			img = Image.fromarray(np_img)
			img = img.resize((input_width, input_height), Image.ANTIALIAS)
		
			img = np.array(img)
			img = img[:, :, ::-1].copy() 
			img = cv2.GaussianBlur(img,(5,5),0)
			im.append(img)
			#numFrames += 1
		minibatch.append(np.stack(im))
		deq.popleft() 

	return minibatch

def broadcast(tensor, shape):
    return tensor + tf.zeros(shape, dtype=tensor.dtype)

def RNNCell(W, B, inputs, state):
	"""Most basic RNN: output = new_state = act(W * input + U * state + B)."""
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = math_ops.tanh

	gate_inputs = math_ops.matmul(array_ops.concat([inputs, state], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, B)
	output = sigmoid(gate_inputs)
	return output, output

def lstm_cell(W, b, forget_bias, inputs, state):
	one = constant_op.constant(1, dtype=dtypes.int32)
	add = math_ops.add
	multiply = math_ops.multiply
	sigmoid = math_ops.sigmoid
	activation = math_ops.sigmoid
	# activation = math_ops.tanh

	c, h = array_ops.split(value=state, num_or_size_splits=2, axis=one)

	gate_inputs = math_ops.matmul(array_ops.concat([inputs, h], 1), W)
	gate_inputs = nn_ops.bias_add(gate_inputs, b)
	# i = input_gate, j = new_input, f = forget_gate, o = output_gate
	i, j, f, o = array_ops.split(value=gate_inputs, num_or_size_splits=4, axis=one)

	forget_bias_tensor = constant_op.constant(forget_bias, dtype=f.dtype)

	new_c = add(multiply(c, sigmoid(add(f, forget_bias_tensor))), multiply(sigmoid(i), activation(j)))
	new_h = multiply(activation(new_c), sigmoid(o))
	new_state = array_ops.concat([new_c, new_h], 1)

	return new_h, new_state
#main func

def lstmAL(miniBatch,q1):

	lr = 1e-10

	activeLearningInput = True #sys.argv[4]


	if activeLearningInput == True:
		activeLearning = True
	else:
		activeLearning = False
	#################to automate
	###################
	#batch = ["./testVideo/C01/C02_403.mp4"] ############

	inputs = tf.placeholder(tf.float32, (None, 224, 224, 3), name='inputs')
	learning_rate = tf.placeholder(tf.float32, [])
	is_training = tf.placeholder(tf.bool)


	# Setup LSTM
	init_state1 = tf.placeholder(tf.float32, [1, 2*n_hidden1], name="State")
	W_lstm1 = vs.get_variable("W1", shape=[feature_size + n_hidden1, 4*n_hidden1])
	b_lstm1 = vs.get_variable("b1", shape=[4*n_hidden1], initializer=init_ops.zeros_initializer(dtype=tf.float32))

	curr_state1 = init_state1


	W_fc1 = tf.Variable(tf.truncated_normal([n_hidden1, feature_size], stddev=0.1))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[feature_size]))

	scope = 'vgg_16'
	fc_conv_padding = 'VALID'
	dropout_keep_prob=0.8

	r, g, b = tf.split(axis=3, num_or_size_splits=3, value=inputs * 255.0)
	VGG_MEAN = [103.939, 116.779, 123.68]
	VGG_inputs = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

	with tf.variable_scope(scope, 'vgg_16', [VGG_inputs]) as sc:
		end_points_collection = sc.original_name_scope + '_end_points'
		# Collect outputs for conv2d, fully_connected and max_pool2d.
		with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
												outputs_collections=end_points_collection):
			net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
			net = slim.max_pool2d(net, [2, 2], scope='pool1')
			net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
			net = slim.max_pool2d(net, [2, 2], scope='pool2')
			net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
			net = slim.max_pool2d(net, [2, 2], scope='pool3')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
			net = slim.max_pool2d(net, [2, 2], scope='pool4')
			net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
			net = slim.max_pool2d(net, [2, 2], scope='pool5')

			# Use conv2d instead of fully_connected layers.
			net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
			net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
												 scope='dropout6')
			net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
			vgg16_Features = tf.reshape(net, (-1,4096))
			# Convert end_points_collection into a end_point dict.
			end_points = slim.utils.convert_collection_to_dict(end_points_collection)

	RNN_inputs = tf.reshape(vgg16_Features[0,:], (-1, feature_size))

	# RNN
	# h_1, curr_state1 = RNNCell(W_RNN1, b_RNN1, RNN_inputs, curr_state1)

	# LSTM
	h_1, curr_state1 = lstm_cell(W_lstm1, b_lstm1, 1.0, RNN_inputs, curr_state1)

	# fc1 = tf.matmul(h_1, W_fc1) + b_fc1
	fc1 = h_1
	print(fc1[0,:].shape, vgg16_Features[1,:].shape)
	sseLoss1 = tf.square(tf.subtract(fc1[0,:], vgg16_Features[1,:]))
	mask = tf.greater(sseLoss1, learnError * tf.ones_like(sseLoss1))
	sseLoss1 = tf.multiply(sseLoss1, tf.cast(mask, tf.float32))
	sseLoss = tf.reduce_mean(sseLoss1)

	# Optimization
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(sseLoss)


	#####################
	### Training loop ###
	#####################

	init = tf.global_variables_initializer()

	saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
	with tf.Session() as sess:
		# Initialize parameters
		sess.run(init)
		#saver.restore(sess, "./vgg_16.ckpt")

		saver = tf.train.Saver(max_to_keep=0)
		avgPredError = 1.0

		### In case of interruption, load parameters from the last iteration (ex: 29)
		#saver.restore(sess, './model_stacked_lstm_29')
		### And update the loop to account for the previous iterations
		#for i in range(29,n_epochs):
		step = 0

		# RNN
		# new_state = np.random.uniform(-0.5,high=0.5,size=(1,n_hidden1))

		# LSTM
		new_state = np.random.uniform(-0.5,high=0.5,size=(1,2*n_hidden1))

		for i in range(n_epochs):
			# Run 1 epoch
			loss = []
			#shuffle(batch)

			#for miniBatchPath in batch:
				
				# RNN
				# new_state = np.random.uniform(-0.5,high=0.5,size=(1,n_hidden1))

				# LSTM
			new_state = np.random.uniform(-0.5,high=0.5,size=(1,2*n_hidden1))

			avgPredError = 0
			#minibatches = loadMiniBatch(miniBatchPath)
			minibatches = loadMiniBatch(miniBatch)
			segCount = 0
			predError = collections.deque(maxlen=30)
			#print('Video:', vidName)
			print('minibatch', len(minibatches))
			for x_train in minibatches:
				segCount += 1
				print('segCount', segCount)
				ret = sess.run([train_op, sseLoss, sseLoss1, curr_state1, fc1], feed_dict = {inputs: x_train, is_training: True, init_state1: new_state, learning_rate:lr})
				new_state = ret[3]

				if activeLearning:
					if ret[1]/avgPredError > 1.5:
						lr = 1e-8
						print('Gating n_steps=', segCount, avgPredError, ret[1])
						# predError.clear()
					else:
						# print('NOT Gating n_steps=', segCount, avgPredError, ret[1])
					 	lr = 1e-10
				predError.append(ret[1])
				avgPredError = np.mean(predError)

		#path = './save/'
		#save_path = saver.save(sess, path)

		##########train done / inference start########

		loss = [] 
		#for miniBatchPath in batch:	
		# LSTM
		new_state = np.random.uniform(-0.5,high=0.5,size=(1,2*n_hidden1))

		avgPredError = 0
		#minibatches = loadMiniBatch(miniBatchPath)
		minibatches = loadMiniBatch(miniBatch)
		'''
		try:
			os.remove(vidName+outFileName)
		except OSError:
			pass
		'''
		#break
		segCount = 0
		#predError = collections.deque(maxlen=20)
		predErrors = [] 
		numSegs = 0
		#print('Video:', vidName)
		prevSeg = 0
		for x_train in minibatches:
			segCount += 1
			ret = sess.run([sseLoss, sseLoss1, curr_state1, fc1], feed_dict = {inputs: x_train, is_training: False, init_state1: new_state, learning_rate:lr})
			new_state = ret[2]
			predErrors.append(ret[0])
			#print(segCount,ret[0])
			'''
			with open(vidName+outFileName, 'a') as outFile:
				print(vidName)
				outFile.write('%d\t%2f\n'%(segCount, ret[0]))
			'''
		predErrors_Ori = predErrors
		predErrors = movingaverage(predErrors, 80)
		predErrors = np.gradient(np.array(predErrors)).tolist()

		predBoundaries = signal.argrelextrema(np.array(predErrors), np.greater, order=int(0.57899*200))[0].tolist()
		#predBoundaries.append(len(gtFrames)-1)

		#outFile = predFile.replace('.txt', '_predBoundaries.txt')
		#with open(outFile, 'w') as of:
		for p in predBoundaries:
			#of.write('%d\n'%p)
			print('Anomaly Boundary Frame:', p)
	print("Finish")
	q1.append(p)