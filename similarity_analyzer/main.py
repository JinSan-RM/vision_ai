#  ======================================================================
#
#  outputData 의 1은 Text, 2는 Image 확정성을 위한 API 및 Flask 까지 고려
#
#  ======================================================================

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
    # ImgPath = ImgCtrl.downloadImage( url )
    in_Img = ImgCtrl.ImgCtrl_module.imageUrlToPixels( input_data )
    mat_Img = ImgCtrl.ImgCtrl_module.imageUrlToPixels( match_data )
    # in_Img, mat_Img = cv2.resize(in_Img), cv2.resize(mat_Img)
    in_results = model( in_Img ) # image 인풋
    mat_results = model( mat_Img )
    print("first mark")
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
    cv2.imwrite(f'/code/Img/{in_sketch}.jpg', in_sketch)
    cv2.imwrite(f'/code/Img/{mat_sketch}.jpg', mat_sketch)
    SSD_value = Similarity.ssim_similarity.ssim_similarity_calculator(in_sketch, mat_sketch)
    print(SSD_value)
    # cv2.imwrite('/code/Img/processed_image.jpg', img)
    # _, encoded_image = cv2.imencode('.jpg', img)
    # image_stream = io.BytesIO(encoded_image.tobytes())
    
    return {"SSIM " : SSD_value}

def resize_image(image, size = (1920, 1080)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)





