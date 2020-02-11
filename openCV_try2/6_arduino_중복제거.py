import cv2
import sys
import math
import cv2 as cv
import numpy as np
import serial

cap = cv2.VideoCapture('video.mp4')  # 카메라 이미지 읽은 객체 생성
arduino = serial.Serial('COM7', 9600)  # 아두이노 포트 주의
a = 0
while cap.isOpened():
    ret, src = cap.read()  # 카메라 프레임 읽기

    src = cv2.resize(src, (640, 360))  # 크기조절
    # 그 뭐냐 관심영역 설정하려면 l_src = src[120:360_행, 0:640_열]
    a_p = a
    # 필터 처리 하는 곳################################################
    filt = np.copy(src)
    cv.imshow("Source", filt)
    filt = cv2.medianBlur(filt, 5)  # median필터 작은 먼지들 제거
    # cv.imshow("median", filt)
    filt = cv2.GaussianBlur(filt, (3, 3), 0)  # 가우시안 블러처리(잡음제거)
    cv.imshow("gausian", filt)
    # =================================================================


    # 케니 엣지 검출 하는 곳###############################################
    # 케니 엣지 검출 - 빛의 영향을 크게 받음
    edge = cv2.Canny(filt, 50, 200, None, 3)
    cv.imshow("edge", edge)
    # =================================================================

    # 흑백-col으로 변환-cdst
    blur = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    cdst = np.copy(blur)

    # 복사
    cdstP = np.copy(cdst)
    # 허프 라인 검출##################################################
    lines = cv.HoughLines(edge, 1, np.pi / 180, 150, None, 0, 0)
    if lines is not None:
        GWGP_SUM = 0
        GWGP_CNT = 0
        GWGM_SUM = 0
        GWGM_CNT = 0
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
            GWG = (a + 0.0000001) / (b + 0.0000001)
            if (GWG > 0):
                GWGP_SUM += GWG
                GWGP_CNT += 1
                # print(GWGP_SUM, GWGP_CNT)
            elif (GWG < 0):
                GWGM_SUM += GWG
                GWGM_CNT += 1
                # print(GWGM_SUM, GWGM_CNT)

        if ((GWGP_CNT != 0) & (GWGM_CNT != 0)):  # 선이 두개다 보일 떄 직진
            GWGP = GWGP_SUM / GWGP_CNT
            GWGM = GWGM_SUM / GWGM_CNT
            # print(GWGP, GWGM)
            DIR = (GWGP + GWGM) / 2
            print(DIR)
            a = 0
            ##cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)
        elif ((GWGP_CNT != 0) & (GWGM_CNT == 0)):  # 왼쪽 선만 보일떄 우회전
            DIR = -1 * GWGP_SUM / GWGP_CNT
            print(DIR)
            a = 1
        elif ((GWGP_CNT == 0) & (GWGM_CNT != 0)):  # 오른쪽 선만 보일떄 좌회전
            DIR = -1 * GWGM_SUM / GWGM_CNT
            print(DIR)
            a = -1
            # -값은 오른쪽으로 가고 +값은 왼쪽으로 간다
        if (a_p != a):
            c = str(a)
            c = c.encode('utf-8')
            arduino.write(c)
            # 허프 P 라인 검출
    linesP = cv.HoughLinesP(edge, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # 이미지 출력

    # cv.imshow("Source", edge)  # 원본
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)  # 허프 라인 1
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)  # 허프 라인 2

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 큐 누르면 화면 꺼짐
        break

cap.release()
cv2.destroyAllWindows()