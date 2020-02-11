import cv2
import sys
import math
import cv2 as cv
import numpy as np
from google.colab.patches import cv2_imshow

cap = cv2.VideoCapture('video2.jpg')  # "파일명.mp4" 가능

def makeBlackImage(image, color=False):
    height, width = image.shape[0], image.shape[1]
    if color is True:
        return np.zeros((height, width, 3), np.uint8)
    else:
        if len(image.shape) == 2:
            return np.zeros((height, width), np.uint8)
        else:
            return np.zeros((height, width, 3), np.uint8)


def fillPolyROI(image, points):
    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]
    mask = makeBlackImage(image)
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, points, ignore_mask_color)
    return mask

def ROI(img):
  img_h, img_w = img.shape[:2]
  region = np.array([[(int(img_h*0.45), int(img_h*0.65)), (int(img_w*0.55), int(img_h*0.65)), (int(img_w*0.9), int(img_h*0.9)),(0,int(img_h*0.9))]], dtype = np.int32)
  mask = fillPolyROI(img,region)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img

def Draw(img, line) :
  result=np.copy(img)
  result = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
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
          cv.line(result, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)
          if(int(int(y0 - 1000 * (a)) - int(y0 + 1000 * (a))) == 0 or int((int(x0 - 1000 * (-b)) - int(x0 + 1000 * (-b)))) == 0) : 
              gradient.append(2)
          else:
              gradient.append((int(y0 - 1000 * (a)) - int(y0 + 1000 * (a)))/ (int(x0 - 1000 * (-b)) - int(x0 + 1000 * (-b))))
  return result

for i in range(0,1):
    src = cv2.imread('test5.jpg')
    gradient=[]
    #src = cv2.resize(src, (640, 360))#크기조절
    #그 뭐냐 관심영역 설정하려면 l_src = src[120:360_행, 0:640_열]

    #필터 처리 하는 곳################################################
    filt = np.copy(src)
    cv2_imshow(filt)
    filt = cv2.medianBlur(filt, 5)  # median필터 작은 먼지들 제거
    #cv.imshow("median", filt)
    filt = cv2.GaussianBlur(filt, (3,3), 0)#가우시안 블러처리(잡음제거)
    cv2_imshow(filt)
    #=================================================================


    #케니 엣지 검출 하는 곳###############################################
    #케니 엣지 검출 - 빛의 영향을 크게 받음
    edge = cv2.Canny(filt, 50, 200, None, 3)
    cv2_imshow(edge)
    #=================================================================
    
    #흑백-col으로 변환-cdst
    cdst = np.copy(edge)

    # 허프 라인 검출##################################################
    cdst=ROI(cdst)
    lines = cv.HoughLines(cdst, 1, np.pi / 180, 150, None, 0, 0)
    cdst=Draw(cdst,lines)

    #이미지 출력
    #cv2_imshow("Source", edge)  # 원본
    cv2_imshow(cdst)  # 허프 라인 1
    print(gradient)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 큐 누르면 화면 꺼짐
        break
 
cap.release()
cv2.destroyAllWindows()