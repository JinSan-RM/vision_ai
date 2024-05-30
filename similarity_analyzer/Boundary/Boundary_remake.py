import cv2
import numpy as np


# 빈 이미지 생성 (회색 배경)
image1 = np.full((768, 1152, 3), 128, dtype=np.uint8)

# 위쪽 사각형
cv2.rectangle(image1, (0, 0), (1152, 192), (0, 0, 0), 10)

# 아래쪽 네 개의 작은 사각형
cv2.rectangle(image1, (0, 192), (283, 768), (180, 180, 180), 10)
cv2.rectangle(image1, (293, 192), (576, 768), (255, 0, 0), 1)
cv2.rectangle(image1, (576, 192), (864, 768), (200, 200, 200), 20)
cv2.rectangle(image1, (864, 192), (1152, 768), (0, 0, 0), 10)

# points = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]], dtype=np.int32)
# cv2.fillConvexPoly(image1, points, (100, 100, 100))

### 패턴 만들기 방식 1 : 점 패턴
pattern = np.ones((image1.shape[0], image1.shape[1], 3), dtype=np.uint8) * 255
pattern[::10, ::20] = 0
image1 = cv2.bitwise_and(image1, pattern)    

### 패턴 만들기 방식 2 : 색있는 상자 대입
points = np.array([[[100, 100], [200, 100], [200, 200], [100, 200]]], dtype=np.int32)
cv2.fillConvexPoly(image1, points, (100, 100, 100))

### copyMakeBorder 테스트
# color = [0, 255, 0]  
# image1 = cv2.copyMakeBorder(image1, 20, 20, 10, 10, cv2.BORDER_CONSTANT, value=color)

# 이미지 저장
cv2.imwrite('/Img/rectangle_image1.jpg', image1)

# 이미지 보여주기
cv2.imshow('Image 1', image1)

cv2.waitKey(0)
cv2.destroyAllWindows()