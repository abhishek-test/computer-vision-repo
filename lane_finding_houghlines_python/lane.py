
import cv2
import numpy as np

from binary import *
from utils import *

frameCount = -1
cap = cv2.VideoCapture("/home/abhishek/Videos/test2.mp4")
pt_list = np.array([[259, 720],[507, 332],[634, 332],[1062, 720]])   # (col, row)


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frameCount += 1

    # preprocess
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binaryImg = getBinaryImage(frame, 150, pt_list)
    edgeImg = cv2.Canny(binaryImg, 30, 150, 3)

    # processing lines
    lines = cv2.HoughLinesP(edgeImg, 2, np.pi/180, 30, 50, 5)

    # find and draw lines using slope intercept form
    frame = slopeIntercept(frame, lines)    

    # display output
    displayOutput = display(frameGray, binaryImg, edgeImg, frame)
    cv2.putText(displayOutput, "Frame: " + str(frameCount), (20,20), 3, 0.5, (0,255,255))

    key = cv2.waitKey(1)

    if (key == ord('p')):
        cv2.waitKey(0)

    if(key == ord('q')):
        break


    cv2.imshow("Frame", frame)
    cv2.waitKey(1)





