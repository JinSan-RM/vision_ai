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
    plt.imshow(diff)
    plt.colorbar()
    plt.title(f'SSIM Difference Map (Score: {ssim_similarity:.4f})')
    plt.savefig('/code/Img/diff.jpg')
    plt.close()

    print("SSIM 유사도:", ssim_similarity)   # -1 ~ 1 사이값
    return ssim_similarity

def feature_matching(img1, img2):
    target_size = (640, 640)
    # list1 = []
    # list2 = []
    # size = []
    # list1 = img1.shape
    # list2 = img2.shape
    # if list1[0] >= list2[0]:
    #     size.append(list1[0])
    # else:
    #     size.append(list2[0])
        
    # if list1[1] >= list2[1]:
    #     size.append(list1[1])
    # else:
    #     size.append(list2[1])
    
    
    # img1 = cv2.resize(img1, (size[0], size[1]))
    # img2 = cv2.resize(img2, (size[0], size[1]))
    img1 = resize_with_aspect_ratio(img1, target_size)
    img2 = resize_with_aspect_ratio(img2, target_size)
    # ORB 파라미터 조정
    scaleFactors = [1.2] #, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    nlevelss = [8] # range(8, 12, 1)
    edgeThresholds = range(31, 50, 1) # 20~50
    patchSizes = range(31, 50, 1) # 20~50
    
    import itertools
    for pair in itertools.product(scaleFactors, nlevelss):
        for edgeThreshold, patchSize in zip(edgeThresholds, patchSizes):
            
            print(f'/code/Img/img_matches_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}')
            
            orb = cv2.ORB_create(nfeatures=500, 
                                scaleFactor=pair[0], 
                                nlevels=pair[1], 
                                edgeThreshold=edgeThreshold, 
                                firstLevel=0, 
                                WTA_K=2, 
                                scoreType=cv2.ORB_HARRIS_SCORE, 
                                patchSize=patchSize, 
                                fastThreshold=20)
            
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            
            # Ensure descriptors are not None and are continuous
            if des1 is not None:
                des1 = np.ascontiguousarray(des1)
            if des2 is not None:
                des2 = np.ascontiguousarray(des2)

            if des1 is None or des2 is None:
                print("No descriptors found in one of the images.")
                continue
                return None, 0
            
            # 인덱스 파라미터 설정 ---①
            # FLANN_INDEX_LSH = 6
            
            # index_params= dict(algorithm = FLANN_INDEX_LSH,
            #                 table_number = 6,
            #                 key_size = 12,
            #                 multi_probe_level = 1)
            
            # search_params=dict(checks=32)
            # matcher = cv2.FlannBasedMatcher(index_params, search_params)
            # matches = matcher.match(kp1, kp2)
            # # 매칭 그리기
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
            cv2.imwrite(f'/code/Img/test1/img_matches_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', img_matches)
    return img_matches, len(matches)


def feature_matching_with_shi_tomasi(img1, img2):
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
    # 이미지를 흑백으로 변환
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi 코너 감지기 설정
    max_corners = 500
    quality_level = 0.01
    min_distance = 5
    block_size = 3
    use_harris_detector = True
    k = 0.04
    
    # Shi-Tomasi 코너 감지
    corners1 = cv2.goodFeaturesToTrack(img1_gray, max_corners, quality_level, min_distance, blockSize=block_size, useHarrisDetector=use_harris_detector, k=k)
    corners2 = cv2.goodFeaturesToTrack(img2_gray, max_corners, quality_level, min_distance, blockSize=block_size, useHarrisDetector=use_harris_detector, k=k)

    # ORB 객체 생성 및 설정
    orb = cv2.ORB_create(edgeThreshold=0, patchSize=5, scoreType=cv2.ORB_HARRIS_SCORE)

    # 특징점으로 변환
    kp1 = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=20) for c in corners1]
    kp2 = [cv2.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=20) for c in corners2]

    # ORB 디스크립터 계산
    kp1, des1 = orb.compute(img1_gray, kp1)
    kp2, des2 = orb.compute(img2_gray, kp2)

    # Ensure descriptors are not None and are continuous
    if des1 is not None:
        des1 = np.ascontiguousarray(des1)
    if des2 is not None:
        des2 = np.ascontiguousarray(des2)

    if des1 is None or des2 is None:
        print("No descriptors found in one of the images.")
        return None, 0
    
    # BFMatcher 객체 생성
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # 매칭 계산
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 매칭 결과 시각화
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


    # 매칭 결과 저장
    cv2.imwrite('/code/Img/img_matches_shi_tomasi.jpg', img_matches)

    # 디버깅 정보 출력
    print(f'시토마시 Number of keypoints in image 1: {len(kp1) if kp1 is not None else 0}')
    print(f'시토마시 Number of keypoints in image 2: {len(kp2) if kp2 is not None else 0}')
    print(f'시토마시 Number of matches: {len(matches)}')

    return img_matches, len(matches)

def divide_image(image, divisions=3):
    h, w = image.shape[:2]
    h_div = h // divisions
    w_div = w // divisions
    blocks = []

    for i in range(divisions):
        for j in range(divisions):
            block = image[i*h_div:(i+1)*h_div, j*w_div:(j+1)*w_div]
            blocks.append(block)
    
    return blocks

# def feature_matching_with_blocks(img1, img2):
#     target_size = (640, 640)
#     img1 = resize_with_aspect_ratio(img1, target_size)
#     img2 = resize_with_aspect_ratio(img2, target_size)
#     # ORB 객체 생성 및 설정
#     orb = cv2.ORB_create(nfeatures=500, scaleFactor=1, nlevels=8, edgeThreshold=0, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=5)
    
#     # 이미지를 9등분으로 분할
#     blocks1 = divide_image(img1)
#     blocks2 = divide_image(img2)
    
#     matches_list = []
#     total_matches = 0
    
#     for block1, block2 in zip(blocks1, blocks2):
#         kp1, des1 = orb.detectAndCompute(block1, None)
#         kp2, des2 = orb.detectAndCompute(block2, None)
        
#         if des1 is not None and des2 is not None:
#             bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#             matches = bf.match(des1, des2)
#             matches = sorted(matches, key=lambda x: x.distance)
#             total_matches += len(matches)
#             matches_list.append((block1, kp1, block2, kp2, matches))
    
#     # 매칭 결과 시각화
#     img_matches_list = []
#     for block1, kp1, block2, kp2, matches in matches_list:
#         img_matches = cv2.drawMatches(block1, kp1, block2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
#         img_matches_list.append(img_matches)
    
#     # 결과 저장
#     for idx, img_matches in enumerate(img_matches_list):
#         cv2.imwrite(f'/code/Img/img_matches_block_{idx+1}.jpg', img_matches)
    
#     # 디버깅 정보 출력
#     print(f'Total number of matches: {total_matches}')
    
def resize_with_aspect_ratio(image, target_size):
    # 원본 이미지 크기
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # 비율을 유지하면서 가장 긴 변을 기준으로 리사이징
    aspect_ratio = w / h
    if aspect_ratio > 1:  # 너비가 높이보다 큰 경우
        new_w = target_w
        new_h = int(target_w / aspect_ratio)
    else:  # 높이가 너비보다 큰 경우
        new_h = target_h
        new_w = int(target_h * aspect_ratio)
    
    # 리사이즈
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # 패딩 추가하여 target_size로 맞추기
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]  # 검은색 패딩
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return new_image