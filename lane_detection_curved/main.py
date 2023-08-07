
import cv2
import numpy as np
from findLanes import *
import time

frameCount = -1
cap = cv2.VideoCapture("/home/abhishek/Downloads/project_video.mp4")

src_pts = np.array([[0, 719],[528, 440],[800, 440],[1279, 719]]).astype(np.float32)  
dst_pts = np.array([[0, 359],[0, 0],[639, 0],[639, 359]]).astype(np.float32)

laneObj = FindLaneLines(src_pts, dst_pts)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frameCount += 1

    time_one = time.time()

    birdViewImg = laneObj.perspectiveTransform(frame)
    binaryImg = laneObj.binarize(birdViewImg)
    laneImg = laneObj.findLanes(binaryImg)  
    laneImg = laneObj.groundView(laneImg)
    frame = laneObj.displayImg(frame, laneImg)

    timeTaken = (int)(1000*(time.time() - time_one))

    cv2.putText(frame, "[FrameCnt ] : " + str(frameCount), (50, 50), 2, 0.7, (0,255,255))
    cv2.putText(frame, "[Time (ms)] : " + str(timeTaken),  (50, 80), 2, 0.7, (0,255,255))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if (key == ord('p')):
        cv2.waitKey(0)

    if(key == ord('q')):
        break

# tflite object detection using opencv