# from PIL import Image
import cv2
from urllib.request import urlopen
from urllib.parse import urlparse
import tensorflow as tf
import numpy as np
from io import BytesIO
import os, requests
    

def downloadImage( url : str ): #함수 변경해야함. 로컬 다운로드랑 Img 경로 지정이랑.

    parsed_url = urlparse(url)                          # URL을 불러와서 경로 저장
    filename = os.path.basename(parsed_url.path)
    ROOT_PATH = os.getcwd() + "/"
    print(ROOT_PATH, "<============ ROOT_PATH")

    image_path = ROOT_PATH + "Img/" + filename[:50] + '.jpeg'           
    print(image_path)
    response = requests.get(url, stream=True)               # 200으로 response 되었을때 저장 시도 아닐때 에러
    if response.status_code == 200:
        with open(image_path, 'wb') as out_file:
            out_file.write(response.content)
            print("Image successfully downloaded: ", image_path)
        return image_path
    else:
        print("Unable to download image. HTTP Response Code: ", response.status_code)
        return None
    
# def imageUrlToPixels( source : str ):
    
#     res = urlopen( source ).read()
#     # Image open
#     img = Image.open(BytesIO(res))
#     img_width, img_height = img.size
#     # test_image = tf.image.resize( test_image, size=(224,224))
#     # test_image = tf.expand_dims(test_image, axis=0)

#     # pixels = np.array(test_image)
#     # print(pixels, "<==========pixels")

#     return img

def imageUrlToPixels(source: str):
    # URL로부터 이미지를 바이트로 다운로드
    resp = urlopen(source)
    image_data = resp.read()
    image_data = np.asarray(bytearray(image_data), dtype="uint8")

    # OpenCV를 이용해 이미지 데이터를 이미지로 변환
    img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # OpenCV는 이미지를 BGR 포맷으로 불러오므로, 필요하다면 RGB로 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img