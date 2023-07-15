import cv2
import numpy as np

def binaryFromGray(frame, thresh, mask_pts):

    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, mask_pts, (1))

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameGray = mask * frameGray
    frameGray = cv2.GaussianBlur(frameGray, (5,5), 0)

    _, binaryWhite = cv2.threshold(frameGray, 150, 255, cv2.THRESH_BINARY)

    return binaryWhite


def binaryFromHSV(frame, thresh, mask_pts):

    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask = cv2.fillConvexPoly(mask, mask_pts, (1))
#
    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #hue,sat,val = cv2.split(hsv) 
#
    #_, binaryHue = cv2.threshold(hue, 30, 255, cv2.THRESH_BINARY)
    #_, binarySat = cv2.threshold(sat, 80, 255, cv2.THRESH_BINARY)
    #_, binaryVal = cv2.threshold(val, 80, 255, cv2.THRESH_BINARY)
#
    #binaryHue = mask*binaryHue
    #binarySat = mask*binarySat
    #binaryVal = mask*binaryVal
#
    #binaryHSV =  cv2.bitwise_or(binaryHue, binarySat, binaryVal)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([11, 106, 150])
    upper = np.array([179, 255, 255])
    hsvMask = cv2.inRange(hsv, lower, upper)
    hsvMask = mask * hsvMask

    return hsvMask


def getBinaryImage(frame, thresh, mask_pts):
    binaryWhite = binaryFromGray(frame, thresh, mask_pts)
    binaryYellow = binaryFromHSV(frame, thresh, mask_pts)

    return cv2.bitwise_or(binaryWhite, binaryYellow)


    