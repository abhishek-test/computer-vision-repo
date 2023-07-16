
import cv2
import numpy as np

from binary import *
from utils import *

frameCount = -1
cap = cv2.VideoCapture("/home/abhishek/Downloads/project_video.mp4")

src_pts = np.array([[0, 719],[528, 440],[800, 440],[1279, 719]]).astype(np.float32)  
dst_pts = np.array([[0, 359],[0, 0],[639, 0],[639, 359]]).astype(np.float32)
homography_Mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
_, inv_homography = cv2.invert(homography_Mat)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frameCount += 1

    ## preprocess
    warpedImg = cv2.warpPerspective(frame, homography_Mat, (640,360))        
    binaryImg = getBinaryImage(warpedImg, 150)    
    laneImg, binaryProcessed = findLanes(binaryImg)

    laneOverlaid = cv2.warpPerspective(laneImg, inv_homography, (1280, 720))
    frame = cv2.addWeighted(frame, 1.0, laneOverlaid, 0.3, 0)
  
    # display output
    displayOutput = display(warpedImg, binaryImg, binaryProcessed, frame)
    cv2.putText(displayOutput, "Frame: " + str(frameCount), (20,20), 3, 0.5, (0,255,255))

    cv2.imshow("Frame", displayOutput)

    key = cv2.waitKey(1)

    if (key == ord('p')):
        cv2.waitKey(0)

    if(key == ord('q')):
        break