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
from sklearn.cluster import DBSCAN



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
    
    
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
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

def feature_matching(img1, img2, in_rect_params, mat_rect_params):
    in_rect_params = len([item for item in in_rect_params if item[0] == 2]) * 4
    mat_rect_params = len([item for item in mat_rect_params if item[0] == 2]) * 4
    # img1 = cv2.resize(img1, (1920, 640))
    # img2 = cv2.resize(img2, (1920, 640))
    print(img1.shape, img2.shape, "여기가 사이즈")
    
    # ORB 파라미터 조정
    scaleFactors = [1.2] #, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    nlevelss = [8] # range(8, 12, 1)
    edgeThresholds = range(31, 40, 1) # 20~50
    patchSizes = range(31, 40, 1) # 20~50
    max_distance= [50, 60, 70, 80, 90, 100]
    ssim_value = ssim_similarity_calculator(img1, img2)
    import itertools
    for pair in itertools.product(scaleFactors, nlevelss, max_distance):
        for edgeThreshold, patchSize in zip(edgeThresholds, patchSizes):
            print(f'/code/Img/img_matches_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}')

            orb = cv2.ORB_create(nfeatures=500, 
                                scaleFactor=pair[0], 
                                nlevels=pair[1], 
                                edgeThreshold=edgeThreshold, 
                                firstLevel=0, 
                                WTA_K=2, 
                                scoreType=cv2.ORB_HARRIS_SCORE, 
                                patchSize=patchSize, 
                                fastThreshold=30)

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)
            des1_manual = np.array([computeRectangleDescriptor(img1, kp, patchSize) for kp in kp1])
            des2_manual = np.array([computeRectangleDescriptor(img2, kp, patchSize) for kp in kp2])
            des1_manual_test = np.array([computeRectangleDescriptor1(img1, kp) for kp in kp1])
            des2_manual_test = np.array([computeRectangleDescriptor1(img2, kp) for kp in kp2])

            matches = custom_bf_match(des1_manual, kp1, des2_manual, kp2, max_distance=pair[2])
            matches = sorted(matches, key=lambda x: x.distance)
            print(f'Number of matches: {len(matches)}')

            img_keypoints1 = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0))
            img_keypoints2 = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0))
            cv2.imwrite(f'/code/Img/keypoints/img_keypoints1_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_keypoints1)
            cv2.imwrite(f'/code/Img/keypoints/img_keypoints2_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_keypoints2)

            img_descriptors1 = img1.copy()
            img_descriptors2 = img2.copy()
            for kp in kp1:
                x, y = kp.pt
                cv2.rectangle(img_descriptors1, (int(x - patchSize / 2), int(y - patchSize / 2)),
                              (int(x + patchSize / 2), int(y + patchSize / 2)), (255, 0, 0), 1)
            cv2.imwrite(f'/code/Img/descriptors/img_descriptors_SF1_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_descriptors1)
            for kp in kp2:
                x, y = kp.pt
                cv2.rectangle(img_descriptors2, (int(x - patchSize / 2), int(y - patchSize / 2)),
                              (int(x + patchSize / 2), int(y + patchSize / 2)), (255, 0, 0), 1)
            cv2.imwrite(f'/code/Img/descriptors/img_descriptors_SF2_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_descriptors2)
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0, 255, 0), matchColor=(0, 0, 255), flags=0)
            # Calculate similarity in_rect_params, mat_rect_params
            num_rectangle_points = min(in_rect_params, mat_rect_params)
            
            num_keypoints = min(len(kp1), len(kp2))
            num_matches = len(matches)
            similarity_ratio = num_matches / num_rectangle_points if num_rectangle_points > 0 else 0
            print(f'Keypoint 1 Number : {len(kp1)}')
            print(f'Keypoint 2 Number : {len(kp2)}')
            print(f'Number of matches: {num_matches}')
            print(f'Similarity ratio: {similarity_ratio:.2f}')
            print(f'SSIM Similarity: {ssim_value:.2f}')
            cv2.imwrite(f'/code/Img/matches/img_matches_custom_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_matches)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1_manual_test, des2_manual_test)
            matches = sorted(matches, key=lambda x: x.distance)
            # print(f'Number of matches: {len(matches)}')

            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0, 255, 0), matchColor=(0, 0, 255), flags=0)
            cv2.imwrite(f'/cod/Img/bf/img_matches_BFMatcher_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', img_matches)

            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=32)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            matches = matcher.match(des1_manual_test, des2_manual_test)

            res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=0)

            cv2.imwrite(f'/code/Img/res/res1_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', res)
            cv2.imwrite(f'/code/Img/res/res2_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD{pair[2]}.jpg', res1)
            Combine_Similarity_Value_by_SSIM_n_ORB = (ssim_value * 0.3) + (similarity_ratio * 0.7)
            # Store results
            results = {}
            key = f'SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}_MD_{pair[2]}'
            results[key] = Combine_Similarity_Value_by_SSIM_n_ORB
            print(f'Combine_Similarity_Value_by_SSIM_n_ORB : {Combine_Similarity_Value_by_SSIM_n_ORB}')
    print(results, 'dict results')

    # Find the highest value and its corresponding key
    max_key = max(results, key=results.get)
    max_value = results[max_key]
    print(f'Highest Combine Similarity Value: {max_value} with key: {max_key}')
    return img_matches, len(matches), similarity_ratio

#=======================================
#  itertools 사용하여 전체 경우의 수 대입
#=======================================

    # import itertools
    # for pair in itertools.product(scaleFactors, nlevelss, max_distance):
    #     for edgeThreshold, patchSize in zip(edgeThresholds, patchSizes):
            
    #         print(f'/code/Img/img_matches_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}')
            
    #         orb = cv2.ORB_create(nfeatures=500, 
    #                             scaleFactor=pair[0], 
    #                             nlevels=pair[1], 
    #                             edgeThreshold=edgeThreshold, 
    #                             firstLevel=0, 
    #                             WTA_K=2, 
    #                             scoreType=cv2.ORB_HARRIS_SCORE, 
    #                             patchSize=patchSize, 
    #                             fastThreshold=30)
            
    #         kp1, des1 = orb.detectAndCompute(img1, None)
    #         kp2, des2 = orb.detectAndCompute(img2, None)
    #         des1_manual = np.array([computeRectangleDescriptor(img1, kp, patchSize) for kp in kp1])
    #         des2_manual = np.array([computeRectangleDescriptor(img2, kp, patchSize) for kp in kp2])
    #         des1_manual_test = np.array([computeRectangleDescriptor1(img1, kp) for kp in kp1])
    #         des2_manual_Test = np.array([computeRectangleDescriptor1(img2, kp) for kp in kp2])
    #         # kp1 = filter_keypoints(kp1)
    #         # kp2 = filter_keypoints(kp2)
    #         # matches = custom_bf_match(des1_manual, des2_manual)
    #         matches = custom_bf_match(des1_manual, kp1, des2_manual, kp2, max_distance=pair[2])
    #         matches = sorted(matches, key=lambda x: x.distance)
    #         print(f'Number of matches: {len(matches)}')
    #         img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0,255,0), matchColor=(0,0,255) ,flags=0)
    #         cv2.imwrite(f'/code/Img/custom/img_matches_custom_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', img_matches)
            
    #         bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
    #         matches = bf.match(des1_manual_test, des2_manual_Test)
    #         matches = sorted(matches, key=lambda x: x.distance)

    #         # 디버깅 정보 출력
    #         print(f'Number of matches: {len(matches)}')

    #         img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, singlePointColor=(0,255,0), matchColor=(0,0,255) ,flags=0)
    #         cv2.imwrite(f'/code/Img/bf/img_matches_BFMatcher_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', img_matches)
    #         # cv2.imwrite(f'/code/Img/custom/img_matches_custom_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', img_matches)
    #         # Ensure descriptors are not None and are continuous
            
    #         # 인덱스 파라미터 설정 ---①
    #         FLANN_INDEX_LSH = 6

    #         index_params= dict(algorithm = FLANN_INDEX_LSH,
    #                         table_number = 6,
    #                         key_size = 12,
    #                         multi_probe_level = 1)

    #         search_params=dict(checks=32)
    #         matcher = cv2.FlannBasedMatcher(index_params, search_params)
    #         matches = matcher.match(des1_manual_test, des2_manual_Test)
    #         # 매칭 그리기
    #         res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
    #                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #         res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, \
    #                     flags=0)

    #         cv2.imwrite(f'/code/Img/res/res1_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', res)
    #         cv2.imwrite(f'/code/Img/res/res2_SF_{pair[0]}_NL_{pair[1]}_ET_{edgeThreshold}_PS_{patchSize}.jpg', res1)


    # return img_matches, len(matches)


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

def computeRectangleDescriptor1(image, keypoint, patch_size=31):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center = keypoint.pt
    angle = keypoint.angle * (np.pi / 180.0)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    half_size = patch_size // 2
    descriptor = np.zeros((patch_size, patch_size), dtype=np.uint8)

    for i in range(patch_size):
        for j in range(patch_size):
            x = (i - half_size) * cos_angle - (j - half_size) * sin_angle + center[0]
            y = (i - half_size) * sin_angle + (j - half_size) * cos_angle + center[1]
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                descriptor[i, j] = image[int(y), int(x)]
            else:
                descriptor[i, j] = 0

    return descriptor.flatten()

def computeRectangleDescriptor(image, keypoint, patch_size):
    # 만약 컬러 이미지라면 그레이스케일로 변환
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    center = keypoint.pt  # 특징점의 중심 좌표
    half_size = patch_size // 2  # 사각형 패치의 절반 크기
    descriptor = np.zeros((patch_size, patch_size), dtype=np.uint8)  # 디스크립터를 저장할 배열

    for i in range(patch_size):
        for j in range(patch_size):
            x = int(center[0] - half_size + i)  # 사각형 패치 내의 x 좌표
            y = int(center[1] - half_size + j)  # 사각형 패치 내의 y 좌표
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:  # 이미지 경계 내에 있는지 확인
                descriptor[i, j] = image[y, x]  # 픽셀 값을 디스크립터에 저장
            else:
                descriptor[i, j] = 0  # 경계 밖이면 0으로 설정

    return descriptor.flatten() 

#=======================================
#  custom_bf_match 방식
#=======================================

# def custom_bf_match(descriptors1, keypoints1, descriptors2, keypoints2, max_distance):
#     matches = []
#     for i in range(descriptors1.shape[0]):
#         best_match_index = -1
#         best_match_distance = float('inf')
#         for j in range(descriptors2.shape[0]):
#             # 특징점 간의 위치 거리 계산
#             pt1 = keypoints1[i].pt
#             pt2 = keypoints2[j].pt
#             location_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
#             if location_distance > max_distance:
#                 continue
            
#             # 해밍 거리 계산
#             distance = np.sum(descriptors1[i] != descriptors2[j]).astype(float)
#             # print(distance, "<========distance\n", best_match_distance,"<=======best_match")
#             if distance < best_match_distance:
#                 best_match_distance = distance
#                 best_match_index = j
#         if best_match_index != -1:
#             matches.append(cv2.DMatch(i, best_match_index, best_match_distance))
#     return matches

def custom_bf_match(descriptors1, keypoints1, descriptors2, keypoints2, max_distance, min_distance=10):
    matches = []
    matched_pts1 = []
    matched_pts2 = []

    for i in range(descriptors1.shape[0]):
        best_match_index = -1
        best_match_distance = float('inf')
        for j in range(descriptors2.shape[0]):
            # 특징점 간의 위치 거리 계산
            pt1 = keypoints1[i].pt
            pt2 = keypoints2[j].pt
            location_distance = np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
            
            if location_distance > max_distance:
                continue
            
            # 해밍 거리 계산
            distance = np.sum(descriptors1[i] != descriptors2[j]).astype(float)
            
            # 이미 매칭된 키포인트와의 거리 확인
            if any(np.sqrt((pt1[0] - mp[0])**2 + (pt1[1] - mp[1])**2) < min_distance for mp in matched_pts1) or \
               any(np.sqrt((pt2[0] - mp[0])**2 + (pt2[1] - mp[1])**2) < min_distance for mp in matched_pts2):
                continue
            
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_index = j
        
        if best_match_index != -1:
            matches.append(cv2.DMatch(i, best_match_index, best_match_distance))
            matched_pts1.append(keypoints1[i].pt)
            matched_pts2.append(keypoints2[best_match_index].pt)
    
    return matches