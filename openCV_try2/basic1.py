import cv2
import sys
import math
import cv2 as cv
import numpy as np


cap = cv2.VideoCapture(0)  # "파일명.mp4" 가능
 
while (True):
    ret, src = cap.read()#파일을 객체로
 
    src = cv2.resize(src, (640, 360))#크기조절

   #그 뭐냐 관심영역 설정하려면 l_src = src[120:360_행, 0:640_열]

    blur = cv2.GaussianBlur(src, (3,3), 0)#가우시안 블러처리(잡음제거)

    img_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)# hsv이미지로 변환한다
    
    # 범위를 정하여 hsv이미지에서 원하는 색 영역을 바이너리 이미지로 생성한다.
    # (가운데와 마지막값은 너무어두워서 검은색에 가까운색과 너무 옅어서 흰색에 가까운색을 제외시키기위해 30으로해야한다.
    lower_blue = (13, 60, 120)
    upper_blue = (33, 110, 200)

    # 앞서 선언한 범위값을 사용하여 바이너리 이미지를 얻는다.(범위내에 있는 픽셀들은 흰색이되고 나머지는 검은색이 된다.)
    colres = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    dst1 = cv.Canny(colres, 50, 200, None, 3)  # src의 케니 엣지 검출
    
    dst = cv2.medianBlur(dst1, 1)  # median필터 작은 먼지들 제거
    
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)#흑백을 컬러로 변환-cdst
    cdstP = np.copy(cdst)  # 복사
 
    lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)  # 허프 라인 검출
 
    if lines is not None:
        for i in range(0, len(lines)):  # 삼각 함수 공식 이용 개 어려움
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
 
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)  # 다른 파라미터로 허프 라인 검출(선이 만들어 질 뻔한 선들)
 
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
 
    cv.imshow("Source", blur)  # 원본
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)  # 허프 라인 1
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)  # 허프 라인 2
 
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 큐 누르면 화면 꺼짐
        break
 
cap.release()
cv2.destroyAllWindows()