# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2 as cv
from os import listdir
from os.path import isfile, join

# initialize the HOG descriptor/person detector
hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

positive_image_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\testing'
onlyfiles = [ f for f in listdir(positive_image_path) if isfile(join(positive_image_path,f)) ]
image = np.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
    image[n] = cv.imread( join(positive_image_path,onlyfiles[n]), 0 )
    image[n] = imutils.resize(image[n], width=min(400, image[n].shape[1]))
    orig = image[n].copy()

	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image[n], winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(image[n], (xA, yA), (xB, yB), (0, 255, 0), 2)

	

	# show the output images
    cv.imshow("Before NMS", np.uint8(orig))
    cv.imshow("After NMS", np.uint8(image[n]))
    cv.waitKey(0)