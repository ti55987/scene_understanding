import sys

import cv2
import numpy as np 

import xml.etree.ElementTree as ET
import pickle


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
#	hog = cv2.HOGDescriptor()
	hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
	hog.setSVMDetector(np.array(svm))
	hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.02}
	print "Setting detector complete"
	cap = cv2.VideoCapture(0)

	while(True):

		ret, frame = cap.read()
		if not ret:
			break

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		print "start detecting"
		found, w = hog.detectMultiScale(gray, 0, (8,8), (32,32), 1.02, 2)
		print "end detecting"
		found_filtered = []
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and inside(r, q):
					break
				else:
					found_filtered.append(r)

		draw_detections(frame, found)
		draw_detections(frame, found_filtered, 3)
		print('%d (%d) found' % (len(found_filtered), len(found)))
		key = cv2.waitKey(10)
		if key == 27:
			cv2.destroyAllWindows()
			break

		cv2.imshow('img', frame)

	cap.release()
