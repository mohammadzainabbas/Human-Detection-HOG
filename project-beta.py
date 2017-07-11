from skimage.feature import hog
from skimage.io import imread
from skimage.transform import pyramid_gaussian
from skimage import color
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from imutils.object_detection import non_max_suppression
#from config import *
import imutils
import glob
import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt 





# 1 - Feature Extraction via HOG    

def Extract_Features():

    # Initialize

    positive_image_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\train\person'
    negative_image_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\train\no_person'
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

    # If feature directories don't exist, create them
    if not os.path.isdir(positive_features_path):
        os.makedirs(positive_features_path)

    # If feature directories don't exist, create them
    if not os.path.isdir(negative_features_path):
        os.makedirs(negative_features_path)

    print ("Calculating the descriptors for the positive samples and saving them")
    for img_path in glob.glob(os.path.join(positive_image_path, "*")):
        #print img_path
        
        img = imread(img_path, as_grey=True)
        if descriptor_type == "HOG":
            feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),block_norm='L1', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        feat_name = os.path.split(img_path)[1].split(".")[0] + ".feat"
        feat_path = os.path.join(positive_features_path, feat_name)
        joblib.dump(feat, feat_path, compress=True)
    print ("Positive features saved in {}".format(positive_features_path))

    print ("Calculating the descriptors for the negative samples and saving them")
    for img_path in glob.glob(os.path.join(negative_image_path, "*")):
        img = imread(img_path, as_grey=True)
        if descriptor_type == "HOG":
            feat = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),block_norm='L1', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)
        feat_name = os.path.split(img_path)[1].split(".")[0] + ".feat"
        feat_path = os.path.join(negative_features_path, feat_name)
    
        joblib.dump(feat, feat_path, compress=True)
    print ("Negative features saved in {}".format(negative_features_path))

    print ("Completed calculating features from training images")


# 2 - Train SVM Classifier

def Train_SVM_Classifier():

    positive_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\person'
    negative_features_path = r'E:\CEME\6th Semester\Digital Image Processing\Project\data\features\no_person'

    # Classifiers supported
    classifier_type = 'Linear_SVM'

    feats = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(positive_features_path,"*.feat")):
        feat = joblib.load(feat_path, mmap_mode=None)
        feats.append(feat)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(negative_features_path,"*.feat")):
        feat = joblib.load(feat_path, mmap_mode=None)
        feats.append(feat)
        labels.append(0)
    print (np.array(feats).shape,len(labels))
    if classifier_type is "Linear_SVM":
        linear_svm_classifier = LinearSVC()
        print ("Training a Linear SVM Classifier")
        linear_svm_classifier.fit(feats, labels)
        # If feature directories don't exist, create them
        if not os.path.isdir(os.path.split(model_path)[0]):
            os.makedirs(os.path.split(model_path)[0])
        joblib.dump(linear_svm_classifier, model_path, compress=True)
        print ("Classifier saved to {}".format(model_path))

        
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
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def Detect_Human(filename):
    img = cv.imread(filename)
    img = imutils.resize(img, width = min(400, img.shape[1]))
    minimum_window_size = (64, 128)
    step_size = (10, 10)
    downscale = 1.25

    linear_svm_classifier = joblib.load(os.path.join(model_path, 'svm.model'), mmap_mode=None)

    #List to store the detections
    detections = []
    #The current scale of the image 
    scale = 0

    for img_scaled in pyramid_gaussian(img, downscale = downscale):
        #The list contains detections at the current scale
        if img_scaled.shape[0] < minimum_window_size[1] or img_scaled.shape[1] < minimum_window_size[0]:
            break
        for (x, y, img_window) in Sliding_Window(img_scaled, minimum_window_size, step_size):
            if img_window.shape[0] != minimum_window_size[1] or img_window.shape[1] != minimum_window_size[0]:
                continue
            img_window = color.rgb2gray(img_window)
            feat = hog(img_window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3),block_norm='L1', visualise=False, transform_sqrt=False,feature_vector=True, normalise=None)

            feat = feat.reshape(1, -1)
            pred = linear_svm_classifier.predict(feat)

            if pred == 1:
                
                if linear_svm_classifier.decision_function(feat) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), linear_svm_classifier.decision_function(feat), 
                    int(minimum_window_size[0] * (downscale**scale)),
                    int(minimum_window_size[1] * (downscale**scale))))
                 

            
        scale += 1

    clone = img.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness = 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print ("Score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    print ("shape, ", pick.shape)

    for (xA, yA, xB, yB) in pick:
        cv.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)
    
    plt.axis("off")
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Raw Detection before Non-Max Suppression")
    plt.show()

    plt.axis("off")
    plt.imshow(cv.cvtColor(clone, cv.COLOR_BGR2RGB))
    plt.title("Final Detections after applying Non-Max Suppression")
    plt.show()

def Test_Folder(foldername):

    filenames = glob.iglob(os.path.join(foldername, '*'))
    for filename in filenames:
        Detect_Human(filename)



# Main

print('main()')

#To extract features
print('1 - Extracting Features')
Extract_Features()

#To train and save SVM model
print('2 - Training Linear SVM model')
Train_SVM_Classifier()

#To test
print('3 - Testing')
foldername = r'E:/CEME/6th Semester/Digital Image Processing/Project/data/testing'
Test_Folder(foldername)