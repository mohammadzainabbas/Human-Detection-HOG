from skimage.feature import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from imutils.object_detection import non_max_suppression
from os import listdir
from os.path import isfile, join
import imutils
import glob
import os
import locale
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 

# 1 - Feature Extraction via HOG    

def Extract_Features():

    # Initialize

    positive_image_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\training\person'
    negative_image_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\training\no_person'
    minimum_window_size = [68, 124]
    step_size = [10, 10]
    orientations = 9
    pixels_per_cell = (8,8)
    cells_per_block = (3,3)
    visualize = False
    normalize = True
    positive_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\person'
    negative_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\no_person'
    model_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\models'
    threshold = .3
    descriptor_type = 'HOG'
    feature = []
    features = []
    temp=[]
    labels = []
    bins=8

    print ("Calculating the descriptors for the positive samples")
    onlyfiles = [ f for f in listdir(positive_image_path) if isfile(join(positive_image_path,f)) ]
    img = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        img[n] = cv.imread( join(positive_image_path,onlyfiles[n]), 0 )
        img[n] = imutils.resize(img[n], width=min(400, img[n].shape[1]))
        temp, hog_image = hog(img[n], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualise=True)

        feat = np.array(np.array_split(np.array(temp).flatten(), bins))
        feat = [np.sum(i) for i in feat]

        features.append(np.mean(feat))
        labels.append(1)
        

    print ("Positive features done")

    print ("Calculating the descriptors for the negative samples")
    onlyfiles = []
    feature = []
    temp = []
    onlyfiles = [ f for f in listdir(negative_image_path) if isfile(join(negative_image_path,f)) ]
    img = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        img[n] = cv.imread( join(negative_image_path,onlyfiles[n]), 0 )
        img[n] = imutils.resize(img[n], width=min(400, img[n].shape[1]))
        temp, hog_image = hog(img[n], orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualise=True)

        feat = np.array(np.array_split(np.array(temp).flatten(), bins))
        feat = [np.sum(i) for i in feat]

        features.append(np.mean(feat))
        labels.append(0)

    print ("Negative features done")

    print ("Completed calculating features from training images")

    return features, labels


# 2 - Train SVM Classifier

def Train_SVM_Classifier(feats, labels):

    positive_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\person'
    negative_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\no_person'
    model_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\models'

    # Classifiers supported
    classifier_type = 'Linear_SVM'

    print (np.array(feats).shape,len(labels))
    if classifier_type is "Linear_SVM":
        linear_svm_classifier = LinearSVC()
        print ("Training a Linear SVM Classifier")
        linear_svm_classifier.fit(feats, labels)
    return linear_svm_classifier

        
# 3 - Detection

def Sliding_Window(image, window_size, step_size):
    '''
    This function returns a patch of the input 'image' of size 
    equal to 'window_size'. The first image returned top-left 
    co-ordinate (0, 0) and are increment in both x and y directions
    by the 'step_size' supplied.
    So, the input parameters are-
    image - Input image
    window_size - Size of Sliding Window 
    step_size - incremented Size of Window
    The function returns a tuple -
    (x, y, im_window)
    '''
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def Detect_Human(filename, linear_svm_classifier):
    
    img = cv.imread(filename, 0)
    
    img = imutils.resize(img, width=min(400, img.shape[1]))
    orign = img.copy()
    bins=8
    features = []

    temp, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(4, 4), visualise=True)

    feat = np.array(np.array_split(np.array(temp).flatten(), bins))
    feat = [np.sum(i) for i in feat]
    features.append(np.mean(feat))

    features = np.reshape(features, (len(features),1))

    pred = linear_svm_classifier.predict(features)


    cvhog = cv.HOGDescriptor()
    cvhog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    (rects, weights) = cvhog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8), scale=1.05)

	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv.rectangle(orign, (x, y), (x + w, y + h), (0, 0, 255), 2)
        

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

	

    
##    plt.axis("off")
##    plt.imshow(orign)
##    plt.title("Raw Detection before Non-Max Suppression")
##    plt.show()

    plt.axis("off")
    plt.imshow(img)
    plt.title("Final Detections after applying Non-Max Suppression")
    plt.show()

def Test_Folder(foldername, classifier):

    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        Detect_Human(filename, classifier)



# Main

print('main()')

#To extract features
print('1 - Extracting Features')
features, labels = Extract_Features()

print(len(features))
print(len(labels))

features = np.reshape(features, (len(features),1))


#To train and save SVM model
print('2 - Training Linear SVM model')
linear_svm_classifier = Train_SVM_Classifier(features, labels)

#To test
print('3 - Testing')
foldername = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\testing'
Test_Folder(foldername, linear_svm_classifier)