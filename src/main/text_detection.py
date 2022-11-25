import numpy as np
import cv2 as cv
from imutils.object_detection import non_max_suppression
from utils import 

def resize_image(image, width, height):
    h, w = image.shape[:2]
    ratio_h = h / height
    ratio_w = w / width

    image = cv.resize(image, (width, height))
    return image, ratio_w, ratio_h

def main(image, width, height, detector, min_confidence):
    image = cv.imread(image)
    origin_image = image.copy()

    # Reisze image
    image, ratio_h, ratio_w = resize_image(image, width, height)

    # Layer used for ROI Regression
    layer_names = ['feature_fusion/Conv_7/Sigmoid',
                   'feature_fusion/concat_3']

    # Load model
    cv.dnn.readNet(detector)

    # Getting result from model
    scores, geometry = for
