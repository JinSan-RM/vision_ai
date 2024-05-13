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
from ImgCrtl import ImgCtrl

app = FastAPI()

@app.get('/')
def mainDef( source : str):
    ImgPixcel = ImgCtrl.imageUrlToPixels( source )
    
    return 


