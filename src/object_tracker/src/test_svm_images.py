import sys

import cv2
import numpy as np 

import xml.etree.ElementTree as ET
import pickle

#read in training data
def readImg(filename, svm):
	hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
	hog.setSVMDetector(np.array(svm))
	hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 2.05}
	print "Setting detector complete"
	ext = '.jpg'
	f = open(filename, 'r')
	for line in f:
		if not line.strip():
			continue
		size, path = line.strip().split(',')
		detect( path, ext, int(size), hog)

	f.close()

def detect( path, ext, sz, hog):

	for i in range (1,sz+1):
		name = path + str(i) + ext
		img = cv2.imread(name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#		I = cv2.resize(gray, (64,64))
		print "start detecting"
		found, w = hog.detectMultiScale(gray, 0, (8,8), (32,32), 2.05, 2)
		print "end detecting"
		found_filtered = []
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and inside(r, q):
					break
				else:
					found_filtered.append(r)

#		draw_detections(img, found)
		draw_detections(img, found_filtered, 3)
		print('%d (%d) found' % (len(found_filtered), len(found)))
		cv2.imshow('img', img)
		cv2.waitKey(0)

#        fd = hog.compute(I)
#        train.append(fd)

def inside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q
	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
	for x, y, w, h in rects:
    # the HOG detector returns slightly larger rectangles than the real objects.
    # so slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__': 
	svm = pickle.load(open(sys.argv[1]))
	readImg(sys.argv[2], svm)
