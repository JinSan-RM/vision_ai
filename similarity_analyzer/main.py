#  ======================================================================
#
#  outputData 의 1은 Text, 2는 Image 확정성을 위한 API 및 Flask 까지 고려
#
#  ======================================================================

import Similarity.FID
import Similarity.findcounter
import Similarity.jaccard_similarity
import Similarity.sift_similarity
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
from io import BytesIO
import os, requests, cv2
import ImgCtrl, AIResult, Boundary, Similarity
from ultralytics import YOLO
import io
app = FastAPI()

model = YOLO('/code/ElementDetect.pt')

@app.get('/')
def mainDef( input_source : str, match_source : str ):
    in_out_data = []
    mat_out_data = []
    
    input_data = input_source
    match_data = match_source
    
    in_Img = ImgCtrl.ImgCtrl_module.imageUrlToPixels( input_data )
    mat_Img = ImgCtrl.ImgCtrl_module.imageUrlToPixels( match_data )
    
    in_results = model( in_Img ) 
    mat_results = model( mat_Img )
    
    for result in in_results:
        data = AIResult.AIResult_module.classbox(result)
        data = AIResult.AIResult_module.process_boxes(data)
        in_out_data.append(data)
        
    for result in mat_results:
        data = AIResult.AIResult_module.classbox(result)
        data = AIResult.AIResult_module.process_boxes(data)
        mat_out_data.append(data)
        
    in_out_data = in_out_data[0]
    mat_out_data = mat_out_data[0]
    
    # in_sketch = Boundary.Boundary_module.ImgBoundarySketch(in_Img, in_out_data)        
    # mat_sketch = Boundary.Boundary_module.ImgBoundarySketch(mat_Img, mat_out_data)
    
    in_sketch = Boundary.Boundary_module.ImgBoundaryRemake(in_Img, in_out_data)        
    mat_sketch = Boundary.Boundary_module.ImgBoundaryRemake(mat_Img, mat_out_data)
    
    #================================
    # SIMM & ORB similarity
    #================================
    
    px_V = Similarity.jaccard_similarity.px_similarity(in_sketch, mat_sketch)
    px_ssim = Similarity.ssim_similarity.ssim_similarity_calculator(in_sketch, mat_sketch)
    px_orb = Similarity.ssim_similarity.feature_matching_with_shi_tomasi(in_sketch, mat_sketch)
    px_FID = Similarity.FID.FID_score(in_sketch, mat_sketch)
    
    # # Detect rectangles
    # rectangles_image1 = Similarity.findcounter.detect_rectangles(in_sketch)
    # rectangles_image2 = Similarity.findcounter.detect_rectangles(mat_sketch)

    # # Get image dimensions
    # image1_height, image1_width = in_Img.shape[:2]
    # image2_height, image2_width = mat_Img.shape[:2]

    # # Create feature vectors
    # feature_vector_image1 = Similarity.findcounter.create_feature_vector(rectangles_image1, image1_width, image1_height)
    # feature_vector_image2 = Similarity.findcounter.create_feature_vector(rectangles_image2, image2_width, image2_height)

    # # Calculate similarities
    # euclidean_dist, cosine_sim = Similarity.findcounter.calculate_similarity(feature_vector_image1, feature_vector_image2)
    # Similarity.findcounter.plot_vectors(feature_vector_image1, feature_vector_image2)
    # Similarity.findcounter.plot_similarity_scores(euclidean_dist, cosine_sim)

    # print(f'Euclidean Distance: {euclidean_dist}')
    # print(f'Cosine Similarity: {cosine_sim}')
    # detect_edges_in = Similarity.findcounter.detect_edges(in_sketch)
    # detect_edges_mat = Similarity.findcounter.detect_edges(mat_sketch)
    # detect_lines_in = Similarity.findcounter.detect_lines(detect_edges_in)
    # detect_lines_mat = Similarity.findcounter.detect_lines(detect_edges_mat)
    # pattern_in = Similarity.findcounter.draw_lines(in_sketch, detect_lines_in)
    # pattern_mat = Similarity.findcounter.draw_lines(mat_sketch, detect_lines_mat)
    # cv2.imwrite('/code/Img/pattern_in.jpg', pattern_in)
    # cv2.imwrite('/code/Img/pattern_mat.jpg', pattern_mat)
    
    # px_block = Similarity.ssim_similarity.feature_matching_with_blocks(in_sketch, mat_sketch)
    # print(px_V)
    # Feature matching
    
    num_matches = 0
    # Resize images to the same size
    # size = (800, 600)
    # processed_image1 = cv2.resize(in_sketch, size)
    # processed_image2 = cv2.resize(mat_sketch, size)

    # Convert to grayscale for SSIM calculation
    # processed_image1_gray = cv2.cvtColor(in_sketch, cv2.COLOR_BGR2GRAY)
    # processed_image2_gray = cv2.cvtColor(mat_sketch, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('/code/Img/in_sketch.jpg', in_sketch)
    cv2.imwrite('/code/Img/mat_sketch.jpg', mat_sketch)
    img_matches, num_matches = Similarity.ssim_similarity.feature_matching(in_sketch, mat_sketch)
    
    if img_matches is not None:
        print(f'Number of matches: {num_matches}')
        # Normalize the ORB match score
        keypoints1 = cv2.ORB_create().detect(in_sketch, None)
        keypoints2 = cv2.ORB_create().detect(mat_sketch, None)
        max_possible_matches = min(len(keypoints1), len(keypoints2))
        orb_similarity = num_matches / max_possible_matches if max_possible_matches > 0 else 0
        print(f'ORB Similarity: {orb_similarity}')
    else:
        num_matches = 0
        orb_similarity = 0
        print("Feature matching could not be performed.")
    # combined_similarity = Similarity.ssim_similarity.calculate_combined_similarity(in_sketch, mat_sketch, (220, 220, 200), (105, 105, 105), weight1=0.9, weight2=0.1, weight_ssim=0.5, weight_orb=0.5)

    # Normalize the ORB match score
    # Assuming maximum number of possible matches is min(number of keypoints in img1, img2)
    keypoints1 = cv2.ORB_create().detect(in_sketch, None)
    keypoints2 = cv2.ORB_create().detect(mat_sketch, None)
    max_possible_matches = min(len(keypoints1), len(keypoints2))
    orb_similarity = num_matches / max_possible_matches if max_possible_matches > 0 else 0
    print(f'ORB Similarity: {orb_similarity}')

    # Combine SSIM and ORB scores
    # Assign weights to SSIM and ORB similarities
    weight_ssim = 0.5
    weight_orb = 0.5
    combined_similarity = (weight_ssim * px_ssim) + (weight_orb * orb_similarity)
    print(f'Combined Similarity: {combined_similarity}')
    return {"value " : px_ssim}





