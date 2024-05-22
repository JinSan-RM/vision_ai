#===================================================================
#
# SSIM 알고리즘 : 밝기/대조/구조적 유사도 비교
#
#===================================================================

import cv2
import numpy as np
from skimage import feature
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim

# image1_path = "similarity_test/torriden_3_blocks.png"
# image2_path = "similarity_test/zgsi_3_blocks.png"

# # 이미지 불러오기
# image1 = cv2.imread(image1_path)
# image2 = cv2.imread(image2_path)

# print("Before Resizing========================")
# print(image1.shape)
# print(image2.shape)

# image1 = cv2.resize(image1, (850, 2400))
# image2 = cv2.resize(image2, (850, 2400))

# print("After Resizing=========================")
# print(image1.shape)
# print(image2.shape)

# # 이미지 변환 (grayscale로 변환)
# image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# #============================
# #           계산            #
# #============================

def ssim_similarity_calculator(image1_resized_n_gray, image2_resized_n_gray):
    list1 = []
    list2 = []
    size = []
    image1_resized_n_gray = cv2.cvtColor(image1_resized_n_gray, cv2.COLOR_BGR2GRAY)
    image2_resized_n_gray = cv2.cvtColor(image2_resized_n_gray, cv2.COLOR_BGR2GRAY)
    list1 = image1_resized_n_gray.shape
    list2 = image2_resized_n_gray.shape
    if list1[0] >= list2[0]:
        size.append(list1[0])
    else:
        size.append(list2[0])
        
    if list1[1] >= list2[1]:
        size.append(list1[1])
    else:
        size.append(list2[1])
    image1_resized_n_gray = cv2.resize(image1_resized_n_gray, (size[0], size[1]))
    image2_resized_n_gray = cv2.resize(image2_resized_n_gray, (size[0], size[1]))
    print(image1_resized_n_gray.shape, "<======size1")
    print(image2_resized_n_gray.shape, "<======size2")
    (ssim_similarity, diff) = ssim(image1_resized_n_gray, image2_resized_n_gray, full=True)
    diff = (diff * 255).astype("uint8")

    print("SSIM 유사도:", ssim_similarity)   # -1 ~ 1 사이값
    return ssim_similarity