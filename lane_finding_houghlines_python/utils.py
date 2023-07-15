import cv2
import numpy as np

def display(frameGray, binaryImg, edgeImg, frame):

    displayMask = np.zeros((720,1280, 3),dtype=np.uint8)

    frameGray = cv2.resize(frameGray, (640, 360))
    edgeImg = cv2.resize(edgeImg, (640, 360))
    binaryImg = cv2.resize(binaryImg, (640, 360))
    frame = cv2.resize(frame, (640, 360))

    frameGray = cv2.cvtColor(frameGray, cv2.COLOR_GRAY2BGR)
    edgeImg = cv2.cvtColor(edgeImg, cv2.COLOR_GRAY2BGR)
    binaryImg = cv2.cvtColor(binaryImg, cv2.COLOR_GRAY2BGR)

    displayMask[0:360, 0:640] = frameGray
    displayMask[0:360, 640:] = binaryImg
    displayMask[360:, 0:640] = edgeImg
    displayMask[360:, 640:] = frame

    return displayMask

def slopeIntercept(frame, lines):

    left_slope = []
    right_slope = []

    left_intercept = []
    right_intercept = []

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]

            x1,y1,x2,y2 = l

            slope = (1.0*(y2-y1))/(1.0*(x2-x1))
            intercept = (y1*1.0 - slope*x1)

            if slope==0.0:
                continue


            if(slope < 0):
                left_slope.append(slope)
                left_intercept.append(intercept)
            else:
                right_slope.append(slope)
                right_intercept.append(intercept)

            left_slope_val = np.median(left_slope, axis=0)
            left_intercept_val = np.median(left_intercept, axis=0)
            right_slope_val = np.median(right_slope, axis=0)
            right_intercept_val = np.median(right_intercept, axis=0)


            #print("Slope: " + str(slope))

        y11 = 300
        y22 = 719

        x11_left = (int)((y11 - left_intercept_val)/(left_slope_val))
        x22_left = (int)((y22 - left_intercept_val)/(left_slope_val))

        x11_right = (int)((y11 - right_intercept_val)/(right_slope_val))
        x22_right = (int)((y22 - right_intercept_val)/(right_slope_val))

        cv2.line(frame, (x11_left, y11), (x22_left, y22), (0,0,255), 3, cv2.LINE_AA)
        cv2.line(frame, (x11_right, y11), (x22_right, y22), (0,0,255), 3, cv2.LINE_AA)  

        return frame


    else:
        return None
