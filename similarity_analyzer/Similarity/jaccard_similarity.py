#===================================================================
#
# Jaccard similarity : 공통 픽셀 개수와 합동 픽셀 개수 비교
#
#===================================================================

import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import jaccard_score
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
def calculate_overlap(image1, image2, lower_bound, upper_bound):
    """
    두 이미지의 특정 색상 범위에 해당하는 영역의 겹침을 계산합니다.

    Parameters:
    image1 (ndarray): 첫 번째 이미지
    image2 (ndarray): 두 번째 이미지
    lower_bound (array): 색상 범위의 하한값 (BGR 형식)
    upper_bound (array): 색상 범위의 상한값 (BGR 형식)

    Returns:
    overlap_area (int): 겹치는 픽셀 수
    overlap_ratio (float): 전체 픽셀 수 대비 겹치는 비율
    """
    # 색상 마스크 생성
    color_mask1 = cv2.inRange(image1, lower_bound, upper_bound)
    color_mask2 = cv2.inRange(image2, lower_bound, upper_bound)
    
    # 겹치는 영역 계산
    overlap = cv2.bitwise_and(color_mask1, color_mask2)
    
    # 겹치는 픽셀 수 계산
    overlap_area = np.sum(overlap > 0)
    
    # 전체 픽셀 수 계산
    total_pixels = image1.shape[0] * image1.shape[1]
    
    # 겹치는 비율 계산
    overlap_ratio = overlap_area / total_pixels
    
    
    return overlap_area, overlap_ratio

def jaccard_similarity_calculator(image1_resized, image2_resized):
    
    # image1_resized = cv2.resize(image1_resized, (1920, 1080))
    # image2_resized = cv2.resize(image2_resized, (1920, 1080))
    list1 = []
    list2 = []
    size = []
    list1 = image1_resized.shape
    list2 = image2_resized.shape
    if list1[0] >= list2[0]:
        size.append(list1[0])
    else:
        size.append(list2[0])
        
    if list1[1] >= list2[1]:
        size.append(list1[1])
    else:
        size.append(list2[1])
        
    image1_resized = cv2.resize(image1_resized, (size[0], size[1]))
    image2_resized = cv2.resize(image2_resized, (size[0], size[1]))   
    
    cv2.imwrite('/code/Img/a7.jpg', image1_resized)
    cv2.imwrite('/code/Img/a8.jpg', image2_resized)
    # image1_resized, image2_resized = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # 바이너리 마스크 생성
    threshold = 128
    binary_mask1 = (image1_resized > threshold).astype(np.uint8)
    binary_mask2 = (image2_resized > threshold).astype(np.uint8)

    # 공통 픽셀 및 합동 픽셀 개수 계산
    common_pixels = np.sum(binary_mask1 & binary_mask2)
    total_pixels = np.sum(binary_mask1 | binary_mask2)

    # Jaccard similarity 계산
    jaccard_similarity = common_pixels / total_pixels

    print("Jaccard similarity:", jaccard_similarity)
    return jaccard_similarity

def px_similarity( Img_orig, Img_mat ):
    
    Img_h, Img_w = Img_mat.shape[:2]
    
    Tp = Img_h * Img_w
    
    Img_orig = cv2.resize(Img_orig, (Img_w, Img_h))
    Img_mat = cv2.resize(Img_mat, (Img_w, Img_h))

    Tl = np.array([115,170,100])
    Tu = np.array([115,170,130])

    # 빨간색 영역의 범위 지정
    Rl = np.array([150,200,135])
    Ru = np.array([150,200,165])

    # 이미지 색 겹치는 영역 계산
    To, ToR = calculate_overlap(Img_orig, Img_mat, Tl, Tu)

    # 텍스트 색 겹치는 영역 계산
    Ro, RoR = calculate_overlap(Img_orig, Img_mat, Rl, Ru)

    print(f"Blue overlap area: {To} pixels ({ToR:.2%} of total pixels {Tp})")
    print(f"Red overlap area: {Ro} pixels ({RoR:.2%} of total pixels {Tp})")
    
    ApxC = (To + Ro) / Tp
    return ApxC