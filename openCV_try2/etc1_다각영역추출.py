import cv2
import imutils
import numpy as np
import joblib

pts = []  # 저장 지점


# 통합：mouse callback function
def draw_roi(event, x, y, flags, param):
    img2 = img.copy()

    if event == cv2.EVENT_LBUTTONDOWN:  # 좌측 마우스로 포인트 선택
        pts.append((x, y))  

    if event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 마우스로 마지막 선택 취소
        pts.pop()  

    if event == cv2.EVENT_MBUTTONDOWN:  # 휠 버튼으로 그리기 끝
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        # 다각형 그리기
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))  # ROI 찾는 데에 사용
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 테이블에 표시되는 이미지에 사용

        show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)

        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)

        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)

    if len(pts) > 0:
        # 테이블에 표시된 그림은 다음의 마지막 지점을 그림
        cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

    if len(pts) > 1:
        # 선 그리기
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y 는 마우스 클릭한 위치의 좌표임
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

    cv2.imshow('image', img2)


# 콜백 기능으로 이미지 창, 바인딩 창 만들기
img = cv2.imread("day")
img = imutils.resize(img, width=500)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)
print("[INFO] 왼쪽 클릭 : 지점 선택, 오른클릭 : 마지막 포인트 제거, 휠클릭 : ROI영역 결정 ")
print("[INFO] S를 눌러서 선택 영역을 확장")
print("[INFO] ESC로 종료")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        saved_data = {
            "ROI": pts
        }
        joblib.dump(value=saved_data, filename="config.pkl")
        print("[INFO]ROI가 저장되었습니다.")
        break
cv2.destroyAllWindows()
