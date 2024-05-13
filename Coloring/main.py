    # =============================
    #
    # *Test Desription
    #
    # =============================

from fastapi import FastAPI
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO
import os, requests, cv2
import ImgCrtl

app = FastAPI()

@app.get('/')
def mainDef( source : str ):
    source = source
    url = source
    ImgPath = ImgCrtl.downloadImage( url )
    ImgPixcel = ImgCrtl.imageUrlToPixels( source )
    
    return 


