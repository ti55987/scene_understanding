
Installation:

-opencv
	http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
	Please use opencv 2.XXX


Data:
	Training data format:
		size_1, class_1
		size_2, class_2
		...

Running Command:
	Training classifers, cross validation and generate training model(.pickle file)
		python train_object_classifier.py train.txt svm

	For real time testing
		python test_classifier.py svm.pickle

	For testing normal images
		python test_svm_image.py svm.pickle test.txt




