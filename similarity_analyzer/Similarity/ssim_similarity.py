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
import matplotlib.pyplot as plt
import matplotlib

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

def ssim_similarity_calculator(image1, image2):
    list1 = []
    list2 = []
    size = []
    list1 = image1.shape
    list2 = image2.shape
    if list1[0] >= list2[0]:
        size.append(list1[0])
    else:
        size.append(list2[0])
        
    if list1[1] >= list2[1]:
        size.append(list1[1])
    else:
        size.append(list2[1])
    
    
    image1 = cv2.resize(image1, (size[0], size[1]))
    image2 = cv2.resize(image2, (size[0], size[1]))
    cv2.imwrite('/code/Img/a3.jpg', image1)
    cv2.imwrite('/code/Img/a4.jpg', image2)
    
    
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    print(image1.shape, "<======size1")
    print(image2.shape, "<======size2")
    (ssim_similarity, diff) = ssim(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")
    # matplotlib의 출력을 Agg로 설정하여 GUI 백엔드를 사용하지 않음
    matplotlib.use('Agg')

    # 유사성 맵을 이미지 파일로 저장
    plt.figure(figsize=(6, 3))
    plt.imshow(diff, cmap='gray')
    plt.colorbar()
    plt.title(f'SSIM Difference Map (Score: {ssim_similarity:.4f})')
    plt.savefig('/code/Img/diff.jpg')
    plt.close()

    print("SSIM 유사도:", ssim_similarity)   # -1 ~ 1 사이값
    return ssim_similarity

def feature_matching(img1, img2):
    list1 = []
    list2 = []
    size = []
    list1 = img1.shape
    list2 = img2.shape
    if list1[0] >= list2[0]:
        size.append(list1[0])
    else:
        size.append(list2[0])
        
    if list1[1] >= list2[1]:
        size.append(list1[1])
    else:
        size.append(list2[1])
    
    
    img1 = cv2.resize(img1, (size[0], size[1]))
    img2 = cv2.resize(img2, (size[0], size[1]))
    # ORB 파라미터 조정
    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=20, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Ensure descriptors are not None and are continuous
    if des1 is not None:
        des1 = np.ascontiguousarray(des1)
    if des2 is not None:
        des2 = np.ascontiguousarray(des2)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return None, 0
    
    # 인덱스 파라미터 설정 ---①
    FLANN_INDEX_LSH = 6
    
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6,
                    key_size = 12,
                    multi_probe_level = 1)
    
    search_params=dict(checks=32)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    matches = matcher.match(des1, des2)
    # 매칭 그리기
    # res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
    #             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # cv2.imwrite('/code/Img/res1.jpg', res)
    # 디버깅 정보 출력
    print(f'Number of keypoints in image 1: {len(kp1) if kp1 is not None else 0}')
    print(f'Number of keypoints in image 2: {len(kp2) if kp2 is not None else 0}')

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 디버깅 정보 출력
    print(f'Number of matches: {len(matches)}')

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('/code/Img/img_matches.jpg', img_matches)
    return img_matches, len(matches)


# def extract_color_mask(img, color):
#     # Define range for the color (example for blue color)
#     lower_bound = np.array([color[0] - 10, color[1] - 10, color[2] - 10])
#     upper_bound = np.array([color[0] + 10, color[1] + 10, color[2] + 10])
#     mask = cv2.inRange(img, lower_bound, upper_bound)
#     return mask

# def feature_matching(img1, img2, color1, color2, weight1=0.9, weight2=0.1):
#     # ORB 파라미터 조정
#     orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)

#     # Extract masks for the given colors
#     mask1_img1 = extract_color_mask(img1, color1)
#     mask2_img1 = extract_color_mask(img1, color2)
#     mask1_img2 = extract_color_mask(img2, color1)
#     mask2_img2 = extract_color_mask(img2, color2)

#     # Extract keypoints and descriptors for each color mask
#     kp1_color1, des1_color1 = orb.detectAndCompute(img1, mask1_img1)
#     kp1_color2, des1_color2 = orb.detectAndCompute(img1, mask2_img1)
#     kp2_color1, des2_color1 = orb.detectAndCompute(img2, mask1_img2)
#     kp2_color2, des2_color2 = orb.detectAndCompute(img2, mask2_img2)

#     # Ensure descriptors are not None and are continuous
#     if des1_color1 is not None:
#         des1_color1 = np.ascontiguousarray(des1_color1)
#     if des1_color2 is not None:
#         des1_color2 = np.ascontiguousarray(des1_color2)
#     if des2_color1 is not None:
#         des2_color1 = np.ascontiguousarray(des2_color1)
#     if des2_color2 is not None:
#         des2_color2 = np.ascontiguousarray(des2_color2)

#     # Initialize matcher
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#     # Match descriptors for each color and calculate weighted matches
#     matches_color1 = []
#     if des1_color1 is not None and des2_color1 is not None:
#         matches_color1 = bf.match(des1_color1, des2_color1)
#         matches_color1 = sorted(matches_color1, key=lambda x: x.distance)

#     matches_color2 = []
#     if des1_color2 is not None and des2_color2 is not None:
#         matches_color2 = bf.match(des1_color2, des2_color2)
#         matches_color2 = sorted(matches_color2, key=lambda x: x.distance)

#     # Calculate the number of matches for each color with weights
#     num_matches_color1 = len(matches_color1)
#     num_matches_color2 = len(matches_color2)
#     weighted_num_matches = weight1 * num_matches_color1 + weight2 * num_matches_color2

#     # Draw matches for debugging
#     img_matches_color1 = cv2.drawMatches(img1, kp1_color1, img2, kp2_color1, matches_color1, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     img_matches_color2 = cv2.drawMatches(img1, kp1_color2, img2, kp2_color2, matches_color2, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     cv2.imwrite('/code/Img/res1_color1.jpg', img_matches_color1)
#     cv2.imwrite('/code/Img/res2_color2.jpg', img_matches_color2)

#     return weighted_num_matches

# def calculate_combined_similarity(img1, img2, color1, color2, weight1=0.95, weight2=0.05, weight_ssim=0., weight_orb=0.5):
#     ssim_similarity = ssim_similarity_calculator(img1, img2)

#     orb_similarity = feature_matching(img1, img2, color1, color2, weight1, weight2)

#     keypoints_img_1 = cv2.ORB_create().detect(img1, None)
#     keypoints_img_2 = cv2.ORB_create().detect(img2, None)
#     max_possible_matches = min(len(keypoints_img_1), len(keypoints_img_2))
#     orb_similarity_normalized = orb_similarity / max_possible_matches if max_possible_matches > 0 else 0

#     combined_similarity = (weight_ssim * ssim_similarity) + (weight_orb * orb_similarity_normalized)
#     print(f'Combined Similarity: {combined_similarity}')

#     return combined_similarity