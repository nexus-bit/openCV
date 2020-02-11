import cv2
import sys
import math
import cv2 as cv
import numpy as np
 
cap = cv2.VideoCapture('video2.mp4')#0이면 카메라
 
while (True):
    ret, src = cap.read()#파일을 객체로
 
    src = cv2.resize(src, (640, 360))#크기조절

    blur = cv2.GaussianBlur(src, (7,7), 0)#블러처리해서 케니 엣지에 도움을 줌
    dst = cv.Canny(blur, 50, 200, None, 3)#src의 케니 엣지 검출
    
 
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)#흑백-col으로 변환-cdst
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
 
    cv.imshow("Source", blur)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
