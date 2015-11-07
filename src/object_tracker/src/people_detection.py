#from skimage import measure
#from svmutil import *
import cv2
import numpy as np 

def inside(r, q):
	rx, ry, rw, rh = r
	qx, qy, qw, qh = q
	return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
	for x, y, w, h in rects:
    # the HOG detector returns slightly larger rectangles than the real objects.
    # so we slightly shrink the rectangles to get a nicer output.
		pad_w, pad_h = int(0.15*w), int(0.05*h)
		cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)

if __name__ == '__main__': 
	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
	hogParams = {'winStride': (8, 8), 'padding': (32, 32), 'scale': 1.05}

	cap = cv2.VideoCapture(0)

	while(True):

		ret, frame = cap.read()
		if not ret:
			break

		found, w = hog.detectMultiScale(frame, **hogParams)
		found_filtered = []
		for ri, r in enumerate(found):
			for qi, q in enumerate(found):
				if ri != qi and inside(r, q):
					break
				else:
					found_filtered.append(r)

		#draw_detections(frame, found)
		draw_detections(frame, found_filtered, 3)
		print('%d (%d) found' % (len(found_filtered), len(found)))
		key = cv2.waitKey(10)
		if key == 27:
			cv2.destroyAllWindows()
			break

		cv2.imshow('img', frame)
#		if cv2.waitKey(1) & 0xFF == ord('q'):
#			break
	
	cap.release()
	cv2.destroyAllWindows()
