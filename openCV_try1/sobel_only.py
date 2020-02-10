import cv2
import sys
import math
import cv2 as cv
import numpy as np
 
cap = cv2.VideoCapture('video2.mp4')#0이면 카메라
 
while (True):
    ret, src = cap.read()#파일을 객체로
 
    src = cv2.resize(src, (640, 360))  #크기조절

    #주황색 컬러 hsv 필터처리한 영상에 sobel edge detection
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)#hsv이미지로 변환한다
    lower_blue = (13, 60, 120)
    upper_blue = (33, 110, 200)
    graye = cv2.inRange(img_hsv, lower_blue, upper_blue)
    #컬러필터 처리결과는 아이러니하게도 흑백입니다

    #sobel함수의 입력은 gray-scale
    img_sobel_x = cv2.Sobel(graye, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)#x축 sobel연산
    img_sobel_y = cv2.Sobel(graye, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)#y축 sobel연산

    dst = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);#이얍!




    #dst = cv.Canny(dst0, 50, 200, None, 3)#src의 케니 엣지 검출_결과는흑백


    
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)#라인에 선 입히려고 컬러로 변환
    cdstP = np.copy(cdst)
 
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
 
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
 
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
 
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("오맂널", src)
    cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
