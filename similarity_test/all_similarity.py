import cv2
import numpy as np
from skimage import feature
from scipy.spatial import distance
from skimage.metrics import structural_similarity as ssim

image1_path = "similarity_test/images/bf_torriden_3_blocks.png"
image2_path = "similarity_test/images/zgsi_3_blocks.png"

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

#===================================================================
#
# SIFT 알고리즘 : 특징점 추출 및 매칭
#
#===================================================================
 
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(image1_gray, None)
kp2, des2 = sift.detectAndCompute(image2_gray, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)

# 매칭 점수 기반 정렬
sorted_matches = sorted(matches, key=lambda x: x.distance)

# SIFT 유사도 계산
sift_similarity = len(sorted_matches) / max(len(kp1), len(kp2))

#===================================================================
#
# Jaccard similarity : 공통 픽셀 개수와 합동 픽셀 개수 비교
#
#===================================================================

# 바이너리 마스크 생성
threshold = 128
binary_mask1 = (image1_gray > threshold).astype(np.uint8)
binary_mask2 = (image2_gray > threshold).astype(np.uint8)

# 공통 픽셀 및 합동 픽셀 개수 계산
common_pixels = np.sum(binary_mask1 & binary_mask2)
total_pixels = np.sum(binary_mask1 | binary_mask2)

# Jaccard similarity 계산
jaccard_similarity = common_pixels / total_pixels


#===================================================================
#
# EMD 알고리즘 : 객체 간 거리 계산
#
#===================================================================

# 없는 라이브러리를 사용하라고 난리..!
# # 객체 좌표값 벡터화
# # object_coords1 = [(x, y) for x, y in feature.censure_fast(image1_gray).keypoints]
# # object_coords2 = [(x, y) for x, y in feature.censure_fast(image2_gray).keypoints]

# # object_coords1 = [(x, y) for x, y in feature.CENSURE(image1_gray).keypoints]
# # object_coords2 = [(x, y) for x, y in feature.CENSURE(image2_gray).keypoints]

# # FAST 특징점 추출기 생성
# fast = cv2.FastFeatureDetector_create()

# # 이미지에서 FAST 특징점 검출
# kp1 = fast.detect(image1_gray, None)
# kp2 = fast.detect(image2_gray, None)

# # 좌표 추출 (FAST 특징점은 x, y 좌표만 가지고 있음)
# object_coords1 = [(x.pt[0], x.pt[1]) for x in kp1]
# object_coords2 = [(x.pt[0], x.pt[1]) for x in kp2]

# # 객체 간 거리 계산
# emd_distance = distance.emd(object_coords1, object_coords2)

# # EMD 유사도 계산
# emd_similarity = 1 - (emd_distance / np.sqrt(len(object_coords1) * len(object_coords2)))



# # ============== skimage를 이용하는 방법 : BruteForceMatcher 같은 언급 안된게 갑자기 들어와서 패스

# # 특징점 추출 및 기술자 계산
# detector = ORB()
# descriptors1, keypoints1 = detector.detect_and_compute(image1_gray)
# descriptors2, keypoints2 = detector.detect_and_compute(image2_gray)
# keypoints1 = detector.detect(image1_gray)
# keypoints2 = detector.detect(image2_gray)

# # 특징점 매칭
# matcher = BruteForceMatcher(crossCheck=True)
# matches = matcher.match(descriptors1, descriptors2)

# # EMD 계산
# distance_matrix = np.zeros((len(keypoints1), len(keypoints2)))
# for i, m in enumerate(matches):
#     distance_matrix[m.query, m.train] = np.linalg.norm(keypoints1[m.query] - keypoints2[m.train])

# emd_distance = emd(distance_matrix)

# ============= SIFT를 이용해서 피처 벡터 추출 후 유클리드 거리 계산
# ============= SIFT 특징 : 국소적인 특징점을 기반으로 비교

def calculate_sift_descriptors(image_resized_n_gray):

    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    # 특징점 및 기술자 검출
    keypoints, descriptors = sift.detectAndCompute(image_resized_n_gray, None)
    return descriptors

# SIFT 기술자 계산
descriptors1 = calculate_sift_descriptors(image1_gray)
descriptors2 = calculate_sift_descriptors(image2_gray)

# 평균 기술자 벡터 계산
mean_descriptor1 = np.mean(descriptors1, axis=0)
mean_descriptor2 = np.mean(descriptors2, axis=0)

# 유클리드 거리 계산
euclidean_distance_1 = distance.euclidean(mean_descriptor1, mean_descriptor2)


# ============= HOG를 이용해서 피처 벡터 추출 후 유클리드 거리 계산
# ============= HOG 특징 : 경계선과 구조적 정보를 강조하여 비교
# NOTE : HOG 변환하는게 시간이 좀 오래걸림. 성능 비교 시 참고

# from skimage.feature import hog
# import matplotlib.pyplot as plt

# def calculate_hog(image_resized_n_gray):

#     # HOG 계산
#     hog_descriptor, hog_image = hog(image_resized_n_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True) # 흑백이므로 multichannel argument 불필요
#     return hog_descriptor, hog_image

# # HOG 기술자 계산
# desc1, hog_image1 = calculate_hog(image1_gray)
# desc2, hog_image2 = calculate_hog(image2_gray)

# # 유클리드 거리 계산
# euclidean_distance_2 = distance.euclidean(desc1, desc2)

# # HOG 이미지 시각화
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.title('HOG Image 1')
# plt.imshow(hog_image1, cmap='gray')

# plt.subplot(1, 2, 2)
# plt.title('HOG Image 2')
# plt.imshow(hog_image2, cmap='gray')

# plt.show()

#===================================================================
#
# SSIM 알고리즘 : 밝기/대조/구조적 유사도 비교
#
#===================================================================

for_ssim_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
for_ssim_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
(ssim_similarity, diff) = ssim(for_ssim_image1, for_ssim_image2, full=True)
diff = (diff * 255).astype("uint8")

# 결과 출력
print("SIFT 유사도:", sift_similarity)
print("Jaccard similarity:", jaccard_similarity)
# print("EMD 유사도:", euclidean_distance)
print(f"Euclidean Distance_by_SIFT: {euclidean_distance_1}")
# print(f"Euclidean Distance_by_HOG: {euclidean_distance_2}")
print("SSIM 유사도:", ssim_similarity)   # -1 ~ 1 사이값
