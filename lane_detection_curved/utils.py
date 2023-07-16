import cv2
import numpy as np

def display(warpedImg, binaryImg, binaryImg2, frame):

    displayMask = np.zeros((720,1280, 3),dtype=np.uint8)

    frame = cv2.resize(frame, (640, 360))
    binaryImg = cv2.cvtColor(binaryImg, cv2.COLOR_GRAY2BGR)    
    

    displayMask[0:360, 0:640] = warpedImg
    displayMask[0:360, 640:] = binaryImg
    displayMask[360:, 0:640] = binaryImg2
    displayMask[360:, 640:] = frame

    return displayMask

def fitLane(lx, ly, rx, ry, img):

    try:
        if ( (len(lx)==0) or (len(ly)==0) or (len(rx)==0) or (len(ry)==0) ):
            return img

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
                cv2.circle(img, (xi_l, yi), 5, (0,255,0),-1)
                cv2.circle(img, (xi_r, yi), 5, (0,255,0),-1)
                cv2.line(img, (xi_l, yi), (xi_r, yi), (0,255,0),2)

        # fill polygon road surface
        xbl = (int)(al*359*359 + bl*359 + cl)
        xtl = (int)(al*0*0 + bl*0 + cl)

        xbr = (int)(ar*359*359 + br*359 + cr)
        xtr = (int)(ar*0*0 + br*0 + cr)

        lane_area_pts = np.array([[xbl,359],[xtl,0],[xtr,0],[xbr,359]])
        #img = cv2.fillConvexPoly(img, lane_area_pts, (0,255,0))

        return img
    
    except:
        return img



def findLanes(binaryImg):

    tempImg = binaryImg[300:, 0:640]    
    tempImg = tempImg//255
    histogram = np.sum(tempImg, axis=0).astype(np.uint8)
    
    midPoint = (int)(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midPoint])
    right_base = np.argmax(histogram[midPoint:]) + midPoint

    processedImg = binaryImg.copy()
    msk = processedImg.copy()
    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)

    # display overlay
    binaryImg = cv2.cvtColor(binaryImg, cv2.COLOR_GRAY2BGR)
    cv2.circle(binaryImg, (left_base, 298), 8, (0,0,255), -1)
    cv2.circle(binaryImg, (right_base, 298), 8, (0,0,255), -1)

    for i in range(640):
        binaryImg[298-histogram[i], i] = (0,255,255)    

    #sliding window
    y = 350
    lx = []
    rx = []
    ly = []
    ry = []

    while y>0:

        # left side
        imgL = processedImg[y-35:y, left_base-50:left_base+50]
        left_M = cv2.moments(imgL)        

        if left_M["m00"] != 0:
            lcx = int(left_M["m10"]/left_M["m00"])
            lcy = int(left_M["m01"]/left_M["m00"])
            cv2.circle(msk, (left_base-50 + lcx, y-35+lcy), 3, (255,0,0), -1)

            lx.append(left_base - 50 + lcx)
            ly.append(y - 35 + lcy)
            left_base = left_base-50 + lcx

        # right side
        imgR = processedImg[y-35:y, right_base-50:right_base+50]
        right_M = cv2.moments(imgR)      

        if right_M["m00"] != 0:
            rcx = int(right_M["m10"]/right_M["m00"])
            rcy = int(right_M["m01"]/right_M["m00"])
            cv2.circle(msk, (right_base-50 + rcx, y-35+rcy), 3, (255,0,0), -1)

            rx.append(right_base-50 + rcx)
            ry.append(y-35+rcy)
            right_base = right_base-50 + rcx

        cv2.rectangle(msk, (left_base+50,y), (left_base-50,y-35), (0,255,255), 2)
        cv2.rectangle(msk, (right_base-50,y), (right_base+50,y-35), (0,255,255), 2)

        y -= 35

    laneImg = np.zeros((360,640,3), dtype=np.uint8)
    laneImg = fitLane(lx, ly, rx, ry, laneImg)
    

    return laneImg, msk

