import cv2
import sys
import math
import cv2 as cv
import numpy as np


import serial
print('serial ' + serial.__version__)
PORT = 'COM7'
BaudRate = 9600



ser = serial.Serial(
    port='/dev/cu.usbmodem1421',
    baudrate=9600,
)






        
 
cap = cv2.VideoCapture(0)#0이면 카메라
 
while (True):
    ret, src = cap.read()#파일을 객체로
 
    src = cv2.resize(src, (640, 360))#크기조절

    blur = cv2.GaussianBlur(src, (3,3), 0)#가우시안 블러처리

    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)# hsv이미지로 변환한다
    
    # 범위를 정하여 hsv이미지에서 원하는 색 영역을 바이너리 이미지로 생성한다.
    # (가운데와 마지막값은 너무어두워서 검은색에 가까운색과 너무 옅어서 흰색에 가까운색을 제외시키기위해 30으로해야한다.
    lower_blue = (23-10, 80, 120)
    upper_blue = (23+10, 110, 200)

    # 앞서 선언한 범위값을 사용하여 바이너리 이미지를 얻는다.(범위내에 있는 픽셀들은 흰색이되고 나머지는 검은색이 된다.)
    colres = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    dst = cv.Canny(colres, 50, 200, None, 3)#src의 케니 엣지 검출
    
 
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)#컬러를 흑백으로 변환-cdst
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


    #시리얼통신
    if ser.readable():
    res = ser.readline()
    print(res.decode()[:len(res)-1])


        
    cv.imshow("Source", blur)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()
