import cv2
import sys
import math
import cv2 as cv
import numpy as np
import serial

cap = cv2.VideoCapture('video2.mp4')  # creating camera object
arduino = serial.Serial('COM7', 9600)  # Caution : arduino port check
while cap.isOpened():  # excessive frame calculating
    ret, src = cap.read()  # reading camera 1 frame

    src = cv2.resize(src, (640, 360))  # resize
    # interested area(ROI) setting l_src = src[120:360_row, 0:640_col]


    # filters################################################
    filt = np.copy(src)
    cv.imshow("Source", filt)
    filt = cv2.medianBlur(filt, 5)  # median filter : salt and pepper dust removing
    # cv.imshow("median", filt)
    filt = cv2.GaussianBlur(filt, (3, 3), 0)  # gaussian blur
    #cv.imshow("gausian", filt)
    # =================================================================


    # canny edge detecting###############################################
    edge = cv2.Canny(filt, 50, 200, None, 3)
    #cv.imshow("edge", edge)
    # =================================================================

    # gray to color
    blur = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    cdst = np.copy(blur)

    # copy
    cdstP = np.copy(cdst)
    # hough line detecting##################################################
    lines = cv.HoughLines(edge, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        GWGP_SUM = 0
        GWGP_CNT = 0
        GWGM_SUM = 0
        GWGM_CNT = 0
        for i in range(0, len(lines)):  # Triangle function GONGSIK
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
            GWG = (a + 0.0000001) / (b + 0.0000001)
            if (GWG > 0):
                GWGP_SUM += GWG
                GWGP_CNT += 1
                # print(GWGP_SUM, GWGP_CNT)
            elif (GWG < 0):
                GWGM_SUM += GWG
                GWGM_CNT += 1
                # print(GWGM_SUM, GWGM_CNT)

        if ((GWGP_CNT != 0) and (GWGM_CNT != 0)):  # available watching both line
            GWGP = GWGP_SUM / GWGP_CNT
            GWGM = GWGM_SUM / GWGM_CNT
            # print(GWGP, GWGM)
            DIR = (GWGP + GWGM) / 2
            if( DIR > 0):
                arduino.write(2)
            elif(DIR < 0):
                arduino.write(4)
            else:
                arduino.write(3)
            #arDIR = 90 + int(DIR * 100)
            #print(arDIR)
            #arduino.write(arDIR)

        elif ((GWGP_CNT != 0) and (GWGM_CNT == 0)):  # Can see only left line
            DIR = -1 * GWGP_SUM / GWGP_CNT
            #arDIR = 90 + int(DIR * 100)
            #print(arDIR)
            #arduino.write(arDIR)
            arduino.write(1)
        elif ((GWGP_CNT == 0) and (GWGM_CNT != 0)):  # can see only right line - turn left
            DIR = -1 * GWGM_SUM / GWGM_CNT
            #arDIR = 90 + int(DIR * 100)
            #print(arDIR)
            #arduino.write(arDIR)
            arduino.write(5)
            # - goes right, + goes left
    # image output
    # cv.imshow("Source", edge)  # orgnal
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)  # Huff line printing

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Q to exit
        break

cap.release()
cv2.destroyAllWindows()