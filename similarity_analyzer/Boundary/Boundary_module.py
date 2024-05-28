    #   =================================================
    #
    #   이미지내부의 AI 결과값을 토대로 바운더리 박스 스케치
    #   
    #   스케치된 Boundary Box 
    #
    #   =================================================

import cv2
import numpy as np

def ImgBoundarySketch(img, output_data):
    img_Width, img_Height = img.shape[:2]
    print(img_Width, img_Height, "<=========shape")
    
    # NOTE : 이 부분이 방해되진 않을테지만 체크 필요. 박스 사이즈가 (1920, 1080) 고정이면 더 클 경우 예상을 벗어남
    cv2.rectangle(img, (0, 0), (1920, 1080), (0, 0, 0), thickness = cv2.FILLED)
    if img is None:
        print("로딩 실패")
    for detection in output_data:
        class_id = detection[0]
        start_x = detection[1]
        start_y = detection[2]
        end_x = detection[3]
        end_y = detection[4]
        
        if class_id == 1: # Text Box
            cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (15,15,15), thickness=2)
        elif class_id == 2: # Image Box
            cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (180,180,180), thickness=2)
            # cv2.putText(image, str(class_id), (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    
    return img
    
    
    