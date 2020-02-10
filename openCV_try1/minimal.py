import cv2
import sys
import math
import cv2 as cv
import numpy as np

# hsv필터를 상황에 맞게 변형시킬 수 있음
# 나는 예시 동영상 기준 차선 색깔을 필터링함

cap = cv2.VideoCapture(1)  # 그대로 0으로 바꿔놓면 카메라

while (True):
    ret, src = cap.read()  # 파일-> src

    src = cv2.resize(src, (640, 360))  # 크기조절
    # 카메라가 고정이 되면 인식하는 화면만 자르면 되므로 하늘과 구분 못하는
    # 일은 없을 것이다.
    # 주황색 라인은 관심영역을 왼쪽으로만 잘라서 계산하고
    # 흰색 라인은 관심 영역을 오른쪽으로만 잘라서 계산하면 반대편 차선 고려X

    # 주황색 hsv 필터처리한 영상
    img_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)  # hsv이미지로 변환한다
    lower_blue = (0, 0, 0)
    upper_blue = (240, 100, 60)
    colorfilter = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # 컬러필터 처리결과는 아이러니하게도 흑백임
    # 필터뿐아니라 라인 검출시에도 주황색 따로, 흰색 따로 edge/line검출 할 수 있음

    # sobel edge detect함수의 입력은 gray-scale
    img_sobel_x = cv2.Sobel(colorfilter, cv2.CV_64F, 1, 0, ksize=3)
    img_sobel_x = cv2.convertScaleAbs(img_sobel_x)  # x축 sobel 연산
    img_sobel_y = cv2.Sobel(colorfilter, cv2.CV_64F, 0, 1, ksize=3)
    img_sobel_y = cv2.convertScaleAbs(img_sobel_y)  # y축 sobel 연산

    dst = cv2.addWeighted(img_sobel_x, 1, img_sobel_y, 1, 0);  # xy축 결합

    # cane = cv.Canny(colorfilter, 50, 200, None, 3)#케니 에지는 이제 쓸모없다


    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)  # 라인에 선 입히려고 다시 컬러로 변환
    cdstP = np.copy(cdst)  # 복사
    cdstA = np.copy(cdst)  # 임시

    # 허프 라인 검출 : 180, 150에서 150숫자가 낮아지면 훨씬 민감해진다
    lines = cv.HoughLines(dst, 1, np.pi / 180, 140, None, 0, 0)
    if lines is not None:  # 라인이 있을 경우에만 실행

        for i in range(0, len(lines)):  # 삼각함수 구하는 과정임
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = [int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))]
            pt2 = [int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))]

            # 테스트용 라인그리기 주석 cv.line(cdst, pt1, pt2, (0, 255, 255), 2)
            # 테스트용 변수확인하기 주석 print(rho, " ", theta, " ", a, " ", b, " ", x0, " ", y0, " ", pt1, " ", pt2, "\n")

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (180, 120, 255), 3, cv.LINE_AA)

    cv.imshow("오리지널", src)
    cv.imshow("Source", dst)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


