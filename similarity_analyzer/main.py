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
    
    in_text_boxes = AIResult.AIResult_module.text_box_detector(in_out_data)
    mat_text_boxes = AIResult.AIResult_module.text_box_detector(mat_out_data)
    similarity_text_ratio = Similarity.text_box_similarity_calculator(in_text_boxes, mat_text_boxes)
    
    react_in_sketch = Boundary.Boundary_module.create_patterned_image( rect_params = in_out_data)
    react_mat_sketch = Boundary.Boundary_module.create_patterned_image( rect_params = mat_out_data)
    # cv2.imwrite('/code/Img/react_in_sketch.jpg', react_in_sketch)
    # cv2.imwrite('code/Img/react_mat_sketch.jpg', react_mat_sketch)
    react_ssim = Similarity.ssim_similarity.ssim_similarity_calculator(react_in_sketch, react_mat_sketch)


    print("reacting calculator")
    _, _, similarity_orb_ratio = Similarity.ssim_similarity.feature_matching(react_in_sketch, react_mat_sketch, in_out_data, mat_out_data)
    
    
    weight_txt_boxes = 0.15
    weight_ssim = 0.275
    weight_orb = 0.575 # 0.7
    combined_similarity = 0
    print(f'similarity_text_ratio : {similarity_text_ratio}\nsimilarity_orb_ratio : {similarity_orb_ratio} \nreact_ssim : {react_ssim}')
    combined_similarity = (weight_txt_boxes * similarity_text_ratio) + (weight_ssim * react_ssim) + (weight_orb * similarity_orb_ratio)
    if combined_similarity > 1:
        combined_similarity = 1
    elif combined_similarity < 0:
        combined_similarity = 0 
    print(f'Combined Similarity: {combined_similarity}')
    return {'Combined Similarity:', combined_similarity}





