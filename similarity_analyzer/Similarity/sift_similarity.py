#===================================================================
#
# SIFT 알고리즘 : 특징점 추출 및 매칭
#
#===================================================================
 
import cv2
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

def sift_similarity_calculator(image1_resized_n_gray, image2_resized_n_gray):

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1_resized_n_gray, None)
    kp2, des2 = sift.detectAndCompute(image2_resized_n_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)

    # 매칭 점수 기반 정렬
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # SIFT 유사도 계산
    sift_similarity = len(sorted_matches) / max(len(kp1), len(kp2))

    print("SIFT 유사도:", sift_similarity)
    return sift_similarity