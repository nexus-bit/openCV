############################알오아이    
def ROI(src):  # 사다리꼴 관심 영역
      # Image Size
      img_h = src.shape[0]
      img_w = src.shape[1]
      print(img_h, img_w)
      # Set Region
      region = np.array([[(100, img_h), (440, img_h), (img_w / 2, img_h / 2)]], dtype = np.int32)
      
      # Apply Mask to the Image
      mask = np.zeros_like(src)
      cv2.fillPoly(mask, region, 1)
      masked_img = cv2.bitwise_and(src, mask)
      return masked_img
    blur = np.copy(src)



#########################신기한 실수
포세이돈
def Bird(img) :
  IMAGE_H,IMAGE_W = img.shape[:2]
  img_h,img_w = img.shape[:2]
  src = np.float32([[0, IMAGE_H], [1207, IMAGE_H], [0, 0], [IMAGE_W, 0]])
  dst = np.float32([[(int(img_h*0.45), int(img_h*0.65)), (int(img_w*0.55), int(img_h*0.65)), (int(img_w), int(img_h*1.0)),(0,int(img_h*1.0))]])
  M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
  Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation

  img = np.copy(img)
  img = img[450:(450+IMAGE_H), 0:IMAGE_W] # Apply np slicing for ROI crop
  warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
  cv2_imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results

src = cv2.imread('image.jpg')
#res = cv.resize(src,(640,480))
res = cv2.cvtColor(src, cv.COLOR_BGR2GRAY)#흑백-col으로 변환-cdst
res = cv2.GaussianBlur(res, (3,3), 0)#블러처리해서 케니 엣지에 도움을 줌 // 노이즈 - 가우시안
dst = cv.Canny(res, 100,200)#src의 케니 엣지 검출