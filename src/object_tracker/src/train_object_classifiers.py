#!/usr/bin/env
# train_object_classifiers.py
import sys
import os
#import svm library
#from svmutil import*
import numpy as np
import cv2
import xml.dom.minidom
import xml.etree.ElementTree as ET
import re
import pickle
import random

'''parent class - starting point to add abstraction'''
class StatModel(object):
	def load(self, fn):
		self.model.load(fn)
	def save(self, fn):
		self.model.save(fn)

'''wrapper for OpenCV SimpleVectorMachine algorithm'''
class SVM(StatModel):
	def __init__(self):
		self.model = cv2.SVM()

	def train(self, samples, responses):
    	#setting algorithm parameters
		params = dict( kernel_type = cv2.SVM_LINEAR, 
				svm_type = cv2.SVM_C_SVC,
				C = 1 )
		self.model.train(samples, responses, params = params)

#		self.model.train_auto(samples, responses, None, None, params = params, k_fold=5)

	def predict(self, samples):
		return np.float32( [self.model.predict(s) for s in samples])

# read in training data
def readImg(filename):
	ext = '.jpg'
	f = open(filename, 'r')
	features = []
	for line in f:
		if not line.strip():
			continue
		size, path = line.strip().split(',')
		fd = []
		ComputeFd(fd, path, ext, int(size))
		features.extend(fd)
	
	f.close()
	return (features, int(size))

def ComputeFd(train, path, ext, sz):

	hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
	for i in range (1,sz+1):
		name = path + str(i) + ext
		img = cv2.imread(name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		I = cv2.resize(gray, (64,64))
		fd = hog.compute(I)
		train.append(fd)

# train svm in library 
def trainSVM(data, label, start, end, isWrite):

	if start != end:
		label = [float(-1)]*80
		label[start:end] = [float(1)]*25

	svm = SVM()
	data = np.float32(data)
	responses = np.float32(label)
	
	svm.train(data,responses)	
	if isWrite == True:
		svm.save(sys.argv[2] +".xml")
		writeSVM(sys.argv[2])

	return svm

def writeSVM(filename):
	tree = ET.parse(filename + ".xml")
	root = tree.getroot()
	SVs = root.getchildren()[0].getchildren()[-2].getchildren()[0] 
#	alpha=float(root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[2].text)
	rho = float( root.getchildren()[0].getchildren()[-1].getchildren()[0].getchildren()[1].text )
	svmvec = [float(x) for x in re.sub( '\s+', ' ', SVs.text ).strip().split(' ')]
	svmvec.append(-rho)
	pickle.dump(svmvec, open(filename + ".pickle", 'w'))

# tp/ (tp+fp)
def CalculatePR(predict, label):

	tp = 0.0
	fp = 0.0
	fn = 0.0
	tn = 0.0	
	for i in xrange(0, len(label)):
		if predict[i] == 1:
			if label[i] == 1:
				tp = tp + 1
			else:
				fp = fp + 1
		else:
			if label[i] == 1:
				fn = fn + 1
			else:
				tn = tn + 1

	precision = tp/(tp + fp) if tp != 0 else 0
	recall = tp/(tp + fn) if tp != 0 else 0

	print "Precision: " + str(precision)
	print "Recall: " + str(recall)

def prepare_data(pos,neg,k):
	data = pos + neg
	n = len(data)/k
	index = random.sample(xrange(len(data)), n)
	test = []
	test_l = []
	train = []
	train_l = []
	for i in xrange(0, len(data)):
		if i in index:
			test.append(data[i])
			if i < len(pos):
				test_l.append(float(1))
			else:
				test_l.append(float(-1))
		else:
			train.append(data[i])
			if i < len(pos):
				train_l.append(float(1))
			else:
				train_l.append(float(-1))

	return (train, train_l, test, test_l)

def cross_validation(pos, neg, k):
	
	for i in xrange(0,k):
		train, label, test, test_l = prepare_data(pos, neg, k)
		m = trainSVM(train, label, 0, 0, False)
		res = m.predict(test)
		mask = (res == test_l)
		correct = np.count_nonzero(mask)
		print correct*100.0/res.size
		CalculatePR(res, test_l)

def testSVM(model, start, end, test, sz):
	base = [float(-1)] * 20
	#chair
	test_l = base
	test_l[start:end] = [float(1)]*sz
	res = model.predict(test)
	CalculatePR(res, test_l)


if __name__ == '__main__':
	fd, sz = readImg(sys.argv[1])
	k = 5
	sz = 25
#	test, test_sz = readImg(sys.argv[2])
	label = []
	chair = fd[0:sz]
	human = fd[sz: sz*2]
	monitor = fd[sz*2: sz*3]
	other = fd[sz*3:]
	print "chair cross validation"
	cross_validation(chair, human + monitor + other, k)
	print "human cross validation"
	cross_validation(human, chair + monitor + other , k)
	print "monitor cross validation"
	cross_validation(monitor, chair + human + other, k)
'''
	chair_m = trainSVM(fd, label, 0, 20, False)
	human_m = trainSVM(fd, label, 20, 40, False)
	monitor_m = trainSVM(fd, label, 40, 60, True)
	print "chair PR :"
	testSVM(chair_m, 0, 5, test, test_sz)
	print "human PR :"
	testSVM(human_m, 5, 10, test, test_sz)
	print "monitor PR :"
	testSVM(monitor_m, 10, 15, test, test_sz)

	#m = trainSVM(fd, label, True)
'''
