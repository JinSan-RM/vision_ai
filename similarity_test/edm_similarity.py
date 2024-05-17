#===================================================================
#
# EMD 알고리즘 : 객체 간 거리 계산
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

# ============= SIFT를 이용해서 피처 벡터 추출 후 유클리드 거리 계산
# ============= SIFT 특징 : 국소적인 특징점을 기반으로 비교

def edm_calculator_by_sift(image1_resized_n_gray, image2_resized_n_gray):

    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    # SIFT 기술자 계산
    keypoints1, descriptors1 = sift.detectAndCompute(image1_resized_n_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2_resized_n_gray, None)

    # 평균 기술자 벡터 계산
    mean_descriptor1 = np.mean(descriptors1, axis=0)
    mean_descriptor2 = np.mean(descriptors2, axis=0)

    # 유클리드 거리 계산
    euclidean_distance_1 = distance.euclidean(mean_descriptor1, mean_descriptor2)
    print(f"Euclidean Distance_by_SIFT: {euclidean_distance_1}")
    
    return euclidean_distance_1

# ============= HOG를 이용해서 피처 벡터 추출 후 유클리드 거리 계산
# ============= HOG 특징 : 경계선과 구조적 정보를 강조하여 비교
# NOTE : HOG 변환하는게 시간이 좀 오래걸림. 성능 비교 시 참고

from skimage.feature import hog
import matplotlib.pyplot as plt

def edm_calculator_by_hog(image1_resized_n_gray, image2_resized_n_gray):

    # HOG 계산
    desc1, hog_image1 = hog(image1_resized_n_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    desc2, hog_image2 = hog(image2_resized_n_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # 유클리드 거리 계산
    euclidean_distance_2 = distance.euclidean(desc1, desc2)

    print(f"Euclidean Distance_by_HOG: {euclidean_distance_2}")
    
    return euclidean_distance_2


# ================= 필요시 활성화, 거의 비슷한 이미지만 나옴

# # HOG 이미지 시각화
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('HOG Image 1')
# plt.imshow(hog_image1, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title('HOG Image 2')
# plt.imshow(hog_image2, cmap='gray')

# plt.show()