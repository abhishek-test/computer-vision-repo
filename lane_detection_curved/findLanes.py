import cv2
import numpy as np
from binary import *

class FindLaneLines():

    def __init__(self, src_pts, dst_pts):
        self.binaryThresh = 150
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.isInitialized = False
        self.detectionCount = 0

    def perspectiveTransform(self, frame):
        birdEyeView = cv2.warpPerspective(frame, self.H, (640,360))
        return birdEyeView
    
    def binarize(self, frame):
        binaryImg = getBinaryImage(frame, self.binaryThresh)
        return binaryImg
    
    def findLanes(self, binaryImg):
        img = np.zeros((360,640,3), dtype=np.uint8)

        tempImg = binaryImg[200:, 0:640]    
        tempImg = tempImg//255
        histogram = np.sum(tempImg, axis=0).astype(np.uint8)
    
        midPoint = (int)(histogram.shape[0]/2)
        left_base = np.argmax(histogram[:midPoint])
        right_base = np.argmax(histogram[midPoint:]) + midPoint

        #sliding window
        y = 350
        lx = []
        rx = []
        ly = []
        ry = []

        while y>0:

            # left side
            imgL = binaryImg[y-35:y, left_base-50:left_base+50]
            left_M = cv2.moments(imgL)        

            if left_M["m00"] != 0:
                lcx = int(left_M["m10"]/left_M["m00"])
                lcy = int(left_M["m01"]/left_M["m00"])

                lx.append(left_base - 50 + lcx)
                ly.append(y - 35 + lcy)
                left_base = left_base-50 + lcx

            # right side
            imgR = binaryImg[y-35:y, right_base-50:right_base+50]
            right_M = cv2.moments(imgR)      

            if right_M["m00"] != 0:
                rcx = int(right_M["m10"]/right_M["m00"])
                rcy = int(right_M["m01"]/right_M["m00"])

                rx.append(right_base-50 + rcx)
                ry.append(y-35+rcy)
                right_base = right_base-50 + rcx

            y -= 35

        try:
            al,bl,cl = np.polyfit(ly, lx, 2)
            ar,br,cr = np.polyfit(ry, rx, 2)

            y = np.linspace(0,719,720)    

            xl = al*y*y + bl*y + cl
            xr = ar*y*y + br*y + cr
    
            for i in range(len(y)):
                if(i % 2 == 0):
                    xi_l = (int)(xl[i])
                    xi_r = (int)(xr[i])
                    yi = (int)(y[i])
                    cv2.line(img, (xi_l, yi), (xi_r, yi), (0,255,0),2)

            return img
    
        except:
            return img
        

        
    def groundView(self, birdView):
        _, inv_homography = cv2.invert(self.H)
        laneImg = cv2.warpPerspective(birdView, inv_homography, (1280, 720))
        return laneImg
        
    def displayImg(self, frame, laneImg):
        frame = cv2.addWeighted(frame, 1.0, laneImg, 0.2, 0)
        return frame

