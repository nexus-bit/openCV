import cv2
import numpy as np
from scipy.stats import itemfreq

def get_dominant_color(image, n_colors):  # 지배적인..?컬러 인식하기
    pixels = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
    palette = np.uint8(centroids)
    return palette[np.argmax(itemfreq(labels)[:, -1])]

# 클릭할때까지 사진촬영하기위해 클릭신호 받기
clicked = False
def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

# 카메라 캡쳐하기
cameraCapture = cv2.VideoCapture(0)  # 영상 받아서 읽기, 0:cam
#  cameraCapture = cv2.imread("sign.png", -1);  # 이미지읽기, -1부터 원래영상 컬러 그레이
cv2.namedWindow('camera')
cv2.setMouseCallback('camera', onMouse)

# 루프돌기전에 한프레임 읽기
success, frame = cameraCapture.read()

# 클릭 안하면 무한반복
while success and not clicked:
    cv2.waitKey(1)
    # 한프레임을 루프돌며 계속 읽기
    success, frame = cameraCapture.read()
    # 그레이스케일 이미지로 바꾸기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 메디안 블러처리
    img = cv2.medianBlur(gray, 37)
    # 허프 그라디언트
    # cv2.HoughCircles(8비트 그레이 입력, 검출방법, dp1이면 입력과 동일해상도,
    #                 원과 중심거리(작으면 덜비슷, 크면 원형만) [, circles[,
    #                   매개변수1(캐니에지 검출기에 전달되는 이미지)[, 매개변수2(작으면 오류상승, 크면 검출낮음),
    #                      [, 최소반지름[, 최대반지름]]]]])
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 110, param1=120, param2=40)

    # 원 검출이 되었을 떄
    if not circles is None:
        circles = np.uint16(np.around(circles))
        max_r, max_i = 0, 0
        for i in range(len(circles[:, :, 2][0])):
            if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                max_i = i
                max_r = circles[:, :, 2][0][i]
        x, y, r = circles[:, :, :][0][max_i]
        if y > r and x > r:
            square = frame[y-r:y+r, x-r:x+r]

            dominant_color = get_dominant_color(square, 2)
            if dominant_color[2] > 100:
                # 빨간색이면 stop
                print("STOP")
            elif dominant_color[0] > 80:
                # 그외 색상들
                # 원의 첫 번쨰 존
                zone_0 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                cv2.imshow('Zone0', zone_0)
                zone_0_color = get_dominant_color(zone_0, 1)
                # 원의 두 번쨰 존
                zone_1 = square[square.shape[0]*1//8:square.shape[0]
                                * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                cv2.imshow('Zone1', zone_1)
                zone_1_color = get_dominant_color(zone_1, 1)
                # 원의 두 번쨰 존
                zone_2 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                cv2.imshow('Zone2', zone_2)
                zone_2_color = get_dominant_color(zone_2, 1)

                if zone_1_color[2] < 60:
                    if sum(zone_0_color) > sum(zone_2_color):
                        print("LEFT")
                    else:
                        print("RIGHT")
                else:
                    if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                        print("FORWARD")
                    elif sum(zone_0_color) > sum(zone_2_color):
                        print("FORWARD AND LEFT")
                    else:
                        print("FORWARD AND RIGHT")
            else:
                print("N/A")

        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    cv2.imshow('circles', frame)
    cv2.imshow('camera', frame)


cv2.destroyAllWindows()
cameraCapture.release()
#stop, 좌회전 우회전 직진 좌.직