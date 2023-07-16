import cv2
import numpy as np

def binaryFromGray(frame, thresh):

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = cv2.GaussianBlur(frameGray, (5,5), 0)
    _, binaryWhite = cv2.threshold(frameGray, 150, 255, cv2.THRESH_BINARY)

    return binaryWhite


def binaryFromHSV(frame, thresh):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([17, 0, 0])   # (11,106,150)
    upper = np.array([35, 255, 255])  # (179,255,255)
    hsvMask = cv2.inRange(hsv, lower, upper)

    return hsvMask


def getBinaryImage(frame, thresh):

    binaryWhite = binaryFromGray(frame, thresh)
    binaryYellow = binaryFromHSV(frame, thresh)

    return cv2.bitwise_or(binaryWhite, binaryYellow)


    