    #   =========================================
    #
    #   outputData 의 1은 Text, 2는 Image g
    #   확정성을 위한 API 및 Flask 까지 고려
    #
    #   =========================================

from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import os, requests, cv2
import ImgCtrl, AIResult, Boundary
from ultralytics import YOLO
import io
app = FastAPI()

model = YOLO('/code/ElementDetect.pt')

@app.get('/')
def mainDef( source : str ):
    output_data = []
    url = source
    # ImgPath = ImgCtrl.downloadImage( url )
    Img = ImgCtrl.ImgCtrl_module.imageUrlToPixels( source )
    results = model(Img) # image 인풋
    
    for result in results:
        data = AIResult.AIResult_module.classbox(result)
        data = AIResult.AIResult_module.process_boxes(data)
        output_data.append(data)
    output_data = output_data[0]
        
    img = Boundary.Boundary_module.ImgBoundarySketch(Img, output_data)
    cv2.imwrite('/code/Img/processed_image.jpg', img)
    _, encoded_image = cv2.imencode('.jpg', img)
    image_stream = io.BytesIO(encoded_image.tobytes())
    
    return StreamingResponse(image_stream, media_type="image/jpeg")


