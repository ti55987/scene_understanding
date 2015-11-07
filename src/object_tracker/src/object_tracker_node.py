#!/usr/bin/env python
# object_tracker_node.py
import sys
import cv2
# import svm library
from svmutil import *

from skimage.feature import hog
from skimage import color, exposure, data

import numpy as np

import rospy
from std_msgs.msg import String

def tracker(model):
	# register with ROS, this is a publisher to "object track" topic
	pub = rospy.Publisher('object track', String, queue_size=10)
	rospy.init_node('tracker', anonymous=True)
	rate = rospy.Rate(5)
	cap = cv2.VideoCapture(0)

	# while True:
    # look at the input from camera, detect and classify objects
    # publish any detected objects
    # pause .5 seconds
	while not rospy.is_shutdown():
		ret, frame = cap.read()
		if not ret:
			break
		I = color.rgb2gray(frame)
		fd, hog_image = hog(I, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
		y = [1.0]
		p_label, p_acc, p_val = svm_predict(y, [fd.tolist()], model)
		print p_label
		pub.publish()
		rate.sleep()

if __name__ == '__main__':
	# load trained model
	model = svm_load_model(sys.argv[1])
	try:
		tracker(model)
	except rospy.ROSInterruptException:
		pass

