#  ======================================================================
#
#  outputData 의 1은 Text, 2는 Image 확정성을 위한 API 및 Flask 까지 고려
#
#  ======================================================================

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
    
    in_sketch = Boundary.Boundary_module.ImgBoundarySketch(in_Img, in_out_data)        
    mat_sketch = Boundary.Boundary_module.ImgBoundarySketch(mat_Img, mat_out_data)

    px_V = Similarity.jaccard_similarity.px_similarity(in_sketch, mat_sketch)
    print(px_V)
 
    return {"value " : px_V}





