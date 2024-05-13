
from PIL import Image
from urllib.request import urlopen
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
from io import BytesIO
import os, requests

class ImgCtrl:
    
    def __init__(self) -> None:
        self.url = self
        self.source = self
    
    def downloadImage( url : str ):

        parsed_url = urlparse(url)                          # URL을 불러와서 경로 저장
        filename = os.path.basename(parsed_url.path)

        ROOT_PATH = os.getcwd() + "/"

        image_path = ROOT_PATH + "app/visionai/" + filename           

        response = requests.get(url, stream=True)               # 200으로 response 되었을때 저장 시도 아닐때 에러
        if response.status_code == 200:
            with open(image_path, 'wb') as out_file:
                out_file.write(response.content)
                print("Image successfully downloaded: ", image_path)
            return image_path
        else:
            print("Unable to download image. HTTP Response Code: ", response.status_code)
            return None
        
    def imageUrlToPixels( source : str ):
        
        res = urlopen( source ).read()
        # Image open
        test_image = Image.open(BytesIO(res))

        test_image = tf.image.resize( test_image, size=(224,224))
        test_image = tf.expand_dims(test_image, axis=0)

        pixels = np.array(test_image)

        return pixels