import cv2 as cv
import time
import numpy as np

# https://github.com/immanuvelprathap/OpenCV-Tesseract-EAST-Text-Detector/blob/8d02e1dd980631cceab95d44da8738dac21f1e4e/utils.py
def forward_passer(net, image, layers, timing = True):
