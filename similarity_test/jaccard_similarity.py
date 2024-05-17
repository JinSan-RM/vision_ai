#===================================================================
#
# Jaccard similarity : 공통 픽셀 개수와 합동 픽셀 개수 비교
#
#===================================================================

import cv2
import numpy as np
from scipy.spatial import distance

image1_path = "similarity_test/torriden_3_blocks.png"
image2_path = "similarity_test/zgsi_3_blocks.png"

# 이미지 불러오기
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

print("Before Resizing========================")
print(image1.shape)
print(image2.shape)

image1 = cv2.resize(image1, (850, 2400))
image2 = cv2.resize(image2, (850, 2400))

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

    # 공통 픽셀 및 합동 픽셀 개수 계산
    common_pixels = np.sum(binary_mask1 & binary_mask2)
    total_pixels = np.sum(binary_mask1 | binary_mask2)

    # Jaccard similarity 계산
    jaccard_similarity = common_pixels / total_pixels

    print("Jaccard similarity:", jaccard_similarity)
    return jaccard_similarity