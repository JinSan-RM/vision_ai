#===================================================================
#
# Jaccard similarity : 공통 픽셀 개수와 합동 픽셀 개수 비교
#
#===================================================================
# after opencv process jaccard similarity is broken
import cv2
import numpy as np
from scipy.spatial import distance

image1_path = "similarity_test/images/gened_test_img1.jpeg"
image2_path = "similarity_test/images/gened_test_img1_boxes_thickFILLED.jpeg"

# 이미지 불러오기
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

print("Before Resizing========================")
print(image1.shape)
print(image2.shape)

image1 = cv2.resize(image1, (1000, 1000))
image2 = cv2.resize(image2, (1000, 1000))
# when (850, 2400), jaccard similarity => 0.8328 / with test_img1 & test_img3
# when (1000, 1000), jaccard similarity => 0.8332 / with test_img1 & test_img3
# when (500, 1000), jaccard similarity => 0.8329 / with test_img1 & test_img3

# when (850, 2400), jaccard similarity => 0.99996813 / with test_img2 & test_img3
# when (1000, 1000), jaccard similarity => 0.999944 / with test_img2 & test_img3
# when (500, 1000), jaccard similarity => 0.9999 / with test_img2 & test_img3
### NOTE : 모든 케이스를 해본 것은 아니지만 resize와는 관계가 없는 것으로 보임. 덮는 부분에서 문제가 발생하는 듯
### ==> gened_test_img들을 이용해본 결과 덮는 것 자체는 문제가 없음. 코드 중 백그라운드를 덮는 무언가가 있을 확률 존재
### ==> Qeustion : 그런데 하고보니 기존 이미지가 바이너리화되면서 0, 1이 되는 기준을 알 수 없었음. 바이너리 변환 기준 체크 필요

print("After Resizing=========================")
print(image1.shape)
print(image2.shape)

# 이미지 변환 (grayscale로 변환)
image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#============================
#           계산            #
#============================

def jaccard_similarity_calculator(image1_resized_n_gray, image2_resized_n_gray):

    # 바이너리 마스크 생성
    threshold = 128
    binary_mask1 = (image1_resized_n_gray > threshold).astype(np.uint8)
    binary_mask2 = (image2_resized_n_gray > threshold).astype(np.uint8)

    print("binary_mask1 : ", binary_mask1.shape, binary_mask1)
    print("binary_mask2 : ", binary_mask2.shape, binary_mask2)
    
    # 공통 픽셀 및 합동 픽셀 개수 계산
    common_pixels = np.sum(binary_mask1 & binary_mask2)
    total_pixels = np.sum(binary_mask1 | binary_mask2)

    # Jaccard similarity 계산
    jaccard_similarity = common_pixels / total_pixels

    print("Jaccard similarity:", jaccard_similarity)
    return jaccard_similarity

print(jaccard_similarity_calculator(image1_gray, image2_gray))